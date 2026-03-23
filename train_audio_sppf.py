"""
train_audio_sppf.py -- SPPF Audio Autoencoder Training (v2: FSQ quantization)

Compresses 20ms speech frames (320 samples @ 16kHz) into a compact FSQ-quantized
latent, then reconstructs. Replaces Opus for voice calls.

Architecture:
  Encoder (Conv1d):  [B, 1, 320] -> 4 strided stages -> [B, latent_dim]
  SPPF Core (MLP):   [B, latent_dim] -> [B, latent_dim]
  FSQ Quantizer:     64 floats -> 32 dims x 5 levels = 74 bits/frame = 3.7 kbps
  Decoder (Conv1d):  [B, latent_dim] -> 4 upsample stages -> [B, 1, 320]

Loss:
  MSE (time domain) + Multi-resolution STFT (frequency domain) + Quantization commitment

Dataset: LibriSpeech train-clean-100 / dev-clean (auto-download via torchaudio)

Usage:
  python train_audio_sppf.py --output_dir outputs_audio --num_epochs 200 --batch_size 64
"""

import argparse
import copy
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio


# ─────────────────────────────────────────────
# FSQ (Finite Scalar Quantization)
# Each latent dimension independently rounds to fixed levels.
# No codebook collapse, no dead codes, 97.6% utilization proven in SEM.
# Straight-through estimator for end-to-end training.
#
# 32 dims x 5 levels = 5^32 possible codes (~10^22)
# 32 x log2(5) = ~74 bits/frame = 3.7 kbps at 50 fps
# With DESE suppression (85%): ~0.6 kbps effective
# ─────────────────────────────────────────────

class FSQQuantizer(nn.Module):
    """
    Finite Scalar Quantization — codebook-free, collapse-proof.

    Projects latent_dim down to d_fsq dimensions, rounds each to fixed
    levels, projects back up. Uses iFSQ activation (2*sigmoid(1.6*z)-1)
    for uniform bin utilization.

    Default: 32 dims x 5 levels = 3.7 kbps at 50 fps.
    """

    def __init__(self, latent_dim: int = 64, levels: list = None):
        super().__init__()
        if levels is None:
            levels = [5] * 32  # 32 dims, 5 levels each
        self.levels = levels
        self.d_fsq = len(levels)
        self.latent_dim = latent_dim

        # Project to/from FSQ space
        self.project_down = nn.Linear(latent_dim, self.d_fsq)
        self.project_up = nn.Linear(self.d_fsq, latent_dim)

        # Init project_down with larger std so iFSQ spans all levels
        nn.init.normal_(self.project_down.weight, std=2.0)
        nn.init.zeros_(self.project_down.bias)

        # Register levels as buffer
        self.register_buffer("_levels_t", torch.tensor(levels, dtype=torch.float32))

        # Bits per frame for logging
        self.bits_per_frame = sum(math.ceil(math.log2(lv)) for lv in levels)

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        """iFSQ activation: maps to near-uniform distribution across bins."""
        half = (self._levels_t - 1) / 2  # [2.0, ...] for L=5
        return (2.0 * torch.sigmoid(1.6 * z) - 1.0) * half

    def forward(self, z: torch.Tensor):
        """
        Quantize with straight-through estimator.

        Returns:
            z_q: quantized tensor projected back to latent_dim (STE)
            z: original continuous tensor (for commitment loss)
        """
        # Project down to FSQ space
        z_low = self.project_down(z)  # [B, d_fsq]

        # Bound to quantization range
        z_bounded = self._bound(z_low)

        # Round to nearest integer level
        z_hat = torch.round(z_bounded)

        # Straight-through estimator
        z_hat_st = z_bounded + (z_hat - z_bounded).detach()

        # Project back to latent_dim
        z_q = self.project_up(z_hat_st)  # [B, latent_dim]

        return z_q, z


# ─────────────────────────────────────────────
# ResBlock1d
# ─────────────────────────────────────────────

class ResBlock1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.InstanceNorm1d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.InstanceNorm1d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


# ─────────────────────────────────────────────
# Encoder1d
# Input: [B, 1, 320] → Output: [B, latent_dim]
# 4 Conv1d stages: 1→32→64→128→64
# Each stride=2 halves temporal dim: 320→160→80→40→20
# ─────────────────────────────────────────────

class Encoder1d(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(1, 32, 4, stride=2, padding=1),
            nn.InstanceNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1d(32),
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(32, 64, 4, stride=2, padding=1),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1d(64),
        )
        self.stage3 = nn.Sequential(
            nn.Conv1d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1d(128),
        )
        self.stage4 = nn.Sequential(
            nn.Conv1d(128, 64, 4, stride=2, padding=1),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1d(64),
        )
        # After 4 stages: [B, 64, 20] → flatten → linear
        self.fc = nn.Linear(64 * 20, latent_dim)

    def forward(self, x):
        x = self.stage1(x)   # [B, 32, 160]
        x = self.stage2(x)   # [B, 64, 80]
        x = self.stage3(x)   # [B, 128, 40]
        x = self.stage4(x)   # [B, 64, 20]
        return self.fc(x.flatten(1))  # [B, latent_dim]


# ─────────────────────────────────────────────
# SPPFCore1d — Simple MLP (v1, no temporal state for audio)
# Named `sppf_core` with attribute `net` to match video checkpoint convention.
# ─────────────────────────────────────────────

class SPPFCore1d(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z):
        return self.net(z)


# ─────────────────────────────────────────────
# Decoder1d
# Input: [B, latent_dim] → Output: [B, 1, 320]
# Mirror of encoder: linear → reshape → 4 upsample stages
# ─────────────────────────────────────────────

class Decoder1d(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 20)

        self.stage1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.InstanceNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1d(128),
        )
        self.stage2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1d(64),
        )
        self.stage3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.InstanceNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1d(32),
        )
        self.stage4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(32, 1, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 64, 20)  # [B, 64, 20]
        x = self.stage1(x)   # [B, 128, 40]
        x = self.stage2(x)   # [B, 64, 80]
        x = self.stage3(x)   # [B, 32, 160]
        x = self.stage4(x)   # [B, 1, 320]
        return x


# ─────────────────────────────────────────────
# Full Autoencoder
# ─────────────────────────────────────────────

class SPPFAudioAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder_part = Encoder1d(latent_dim)
        self.sppf_core = SPPFCore1d(latent_dim)
        self.quantizer = FSQQuantizer(latent_dim=latent_dim)
        self.decoder_part = Decoder1d(latent_dim)

    def forward(self, x):
        z = self.encoder_part(x)          # [B, latent_dim]
        z = self.sppf_core(z)             # [B, latent_dim]
        z_q, z_cont = self.quantizer(z)   # quantized (STE), continuous
        recon = self.decoder_part(z_q)    # [B, 1, 320]
        return recon, z_cont, z_q


# ─────────────────────────────────────────────
# Multi-resolution STFT Loss
# ─────────────────────────────────────────────

class MultiResolutionSTFTLoss(nn.Module):
    """
    Compute spectral convergence + log magnitude L1 at multiple FFT sizes.
    Critical for audio quality — MSE alone produces muffled output.
    """

    def __init__(self, fft_sizes=(64, 128, 256)):
        super().__init__()
        self.fft_sizes = fft_sizes

    def _stft(self, x, n_fft):
        """Compute magnitude spectrogram."""
        # x: [B, T]
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=x.device)
        # torch.stft returns complex tensor
        spec = torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=win, return_complex=True,
        )
        return spec.abs()  # magnitude: [B, n_fft//2+1, T_frames]

    def forward(self, x_recon, x_target):
        """
        Args:
            x_recon:  [B, 1, 320]
            x_target: [B, 1, 320]
        """
        x_recon = x_recon.squeeze(1)    # [B, 320]
        x_target = x_target.squeeze(1)  # [B, 320]

        total_loss = 0.0
        for n_fft in self.fft_sizes:
            mag_recon = self._stft(x_recon, n_fft)
            mag_target = self._stft(x_target, n_fft)

            # Spectral convergence: Frobenius norm of difference / Frobenius norm of target
            sc = torch.norm(mag_target - mag_recon, p="fro") / (torch.norm(mag_target, p="fro") + 1e-8)

            # Log magnitude L1
            log_mag_recon = torch.log(mag_recon + 1e-8)
            log_mag_target = torch.log(mag_target + 1e-8)
            lm = F.l1_loss(log_mag_recon, log_mag_target)

            total_loss = total_loss + sc + lm

        return total_loss / len(self.fft_sizes)


# ─────────────────────────────────────────────
# Dataset — LibriSpeech with random 320-sample crops
# ─────────────────────────────────────────────

class LibriSpeechChunks(Dataset):
    """
    Wraps torchaudio LibriSpeech dataset.
    Each __getitem__ returns a random 320-sample chunk normalized to [-1, 1].
    """

    def __init__(self, root, url="train-clean-100", download=True):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=url, download=download
        )
        self.target_sr = 16000
        self.chunk_size = 320  # 20ms @ 16kHz

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sample_rate, *_ = self.dataset[idx]

        # Resample if needed
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
            waveform = resampler(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Normalize to [-1, 1]
        max_val = waveform.abs().max().clamp(min=1e-8)
        waveform = waveform / max_val

        # Random crop of 320 samples
        T = waveform.shape[1]
        if T < self.chunk_size:
            # Pad if too short
            waveform = F.pad(waveform, (0, self.chunk_size - T))
            start = 0
        else:
            start = torch.randint(0, T - self.chunk_size + 1, (1,)).item()

        chunk = waveform[:, start:start + self.chunk_size]  # [1, 320]
        return chunk


# ─────────────────────────────────────────────
# EMA helper
# ─────────────────────────────────────────────

def ema_update(ema_shadow, model, decay=0.999):
    """Update EMA shadow weights."""
    with torch.no_grad():
        for key, param in model.state_dict().items():
            if key in ema_shadow:
                ema_shadow[key].mul_(decay).add_(param, alpha=1 - decay)


def ema_load(model, ema_shadow):
    """Load EMA weights into model for evaluation."""
    model.load_state_dict(ema_shadow)


# ─────────────────────────────────────────────
# SNR computation
# ─────────────────────────────────────────────

def compute_snr(original, reconstructed):
    """Compute signal-to-noise ratio in dB."""
    signal_power = (original ** 2).mean()
    noise_power = ((original - reconstructed) ** 2).mean()
    if noise_power < 1e-12:
        return float('inf')
    return 10 * math.log10(signal_power.item() / noise_power.item())


def compute_pesq_batch(original, reconstructed, sr=16000):
    """Compute PESQ (ITU-T P.862) on a batch. Returns mean score or None."""
    try:
        from pesq import pesq as pesq_fn
    except ImportError:
        return None
    scores = []
    orig_np = original.squeeze(1).cpu().numpy()  # [B, T]
    recon_np = reconstructed.squeeze(1).cpu().numpy()
    for i in range(orig_np.shape[0]):
        try:
            score = pesq_fn(sr, orig_np[i], recon_np[i], 'wb')  # wideband
            scores.append(score)
        except Exception:
            continue
    return sum(scores) / len(scores) if scores else None


# ─────────────────────────────────────────────
# Multi-Scale Waveform Discriminator
# Same idea as the video model's PatchGAN — forces the decoder to
# produce realistic waveforms, not muffled MSE-average output.
# Operates at 2 scales: original (320 samples) and downsampled (160).
# ─────────────────────────────────────────────

class MelSpectrogramLoss(nn.Module):
    """Perceptually-weighted mel-spectrogram L1 loss.
    Mel scale matches human hearing — more important than raw STFT for speech."""

    def __init__(self, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer(
            "mel_basis",
            self._mel_filterbank(sr, n_fft, n_mels)
        )

    @staticmethod
    def _mel_filterbank(sr, n_fft, n_mels):
        """Create mel filterbank matrix."""
        # Simplified mel filterbank (linear in mel space)
        f_min, f_max = 0.0, sr / 2.0
        mel_min = 2595.0 * math.log10(1.0 + f_min / 700.0)
        mel_max = 2595.0 * math.log10(1.0 + f_max / 700.0)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bins = torch.floor((n_fft + 1) * hz_points / sr).long()

        fb = torch.zeros(n_mels, n_fft // 2 + 1)
        for m in range(n_mels):
            f_left, f_center, f_right = bins[m], bins[m + 1], bins[m + 2]
            for k in range(f_left, f_center):
                if f_center != f_left:
                    fb[m, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                if f_right != f_center:
                    fb[m, k] = (f_right - k) / (f_right - f_center)
        return fb

    def forward(self, x_recon, x_target):
        """x_recon, x_target: [B, 1, T]"""
        x_r = x_recon.squeeze(1)
        x_t = x_target.squeeze(1)

        win = torch.hann_window(self.n_fft, device=x_r.device)
        spec_r = torch.stft(x_r, self.n_fft, self.hop_length, window=win, return_complex=True).abs()
        spec_t = torch.stft(x_t, self.n_fft, self.hop_length, window=win, return_complex=True).abs()

        mel_r = torch.log(torch.matmul(self.mel_basis.to(x_r.device), spec_r) + 1e-8)
        mel_t = torch.log(torch.matmul(self.mel_basis.to(x_t.device), spec_t) + 1e-8)

        return F.l1_loss(mel_r, mel_t)


def feature_matching_loss(disc, real, fake):
    """Match intermediate discriminator features (pix2pixHD-style).
    Gives the generator direct texture-level supervision."""
    loss = 0.0
    n_layers = 0
    # Get features from each sub-discriminator
    for sub_disc in [disc.disc1, disc.disc2]:
        feat_real = []
        feat_fake = []
        x_r = real
        x_f = fake
        if sub_disc is disc.disc2:
            x_r = disc.downsample(real)
            x_f = disc.downsample(fake)
        # Extract intermediate features
        for layer in sub_disc.net:
            x_r = layer(x_r)
            x_f = layer(x_f)
            feat_real.append(x_r)
            feat_fake.append(x_f)
        for fr, ff in zip(feat_real[:-1], feat_fake[:-1]):  # skip last (output)
            loss += F.l1_loss(ff, fr.detach())
            n_layers += 1
    return loss / max(n_layers, 1)


def r1_gradient_penalty(disc, real):
    """R1 gradient penalty — stabilizes discriminator (StyleGAN2).
    Penalizes the gradient of D with respect to real samples."""
    real = real.detach().requires_grad_(True)
    outputs = disc(real)
    grad = torch.autograd.grad(
        outputs=[sum(o.sum() for o in outputs)],
        inputs=real,
        create_graph=True,
    )[0]
    return grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()


class WaveformDiscriminator(nn.Module):
    """Single-scale 1D PatchGAN discriminator with spectral norm."""

    def __init__(self):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv1d(1, 32, 15, stride=1, padding=7)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv1d(32, 64, 41, stride=4, padding=20, groups=4)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv1d(64, 128, 41, stride=4, padding=20, groups=16)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv1d(128, 256, 41, stride=4, padding=20, groups=16)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv1d(256, 256, 5, stride=1, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv1d(256, 1, 3, stride=1, padding=1)),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator: original + 2x downsampled."""

    def __init__(self):
        super().__init__()
        self.disc1 = WaveformDiscriminator()
        self.disc2 = WaveformDiscriminator()
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        """Returns list of discriminator outputs at each scale."""
        d1 = self.disc1(x)
        d2 = self.disc2(self.downsample(x))
        return [d1, d2]


def disc_loss_fn(real_outputs, fake_outputs):
    """Hinge loss for discriminator."""
    loss = 0
    for dr, df in zip(real_outputs, fake_outputs):
        loss += torch.mean(F.relu(1 - dr)) + torch.mean(F.relu(1 + df))
    return loss / len(real_outputs)


def gen_loss_fn(fake_outputs):
    """Generator adversarial loss (hinge)."""
    loss = 0
    for df in fake_outputs:
        loss += -torch.mean(df)
    return loss / len(fake_outputs)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ──
    model = SPPFAudioAutoencoder(latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    bits_per_frame = model.quantizer.bits_per_frame
    print(f"\nCompression stats:")
    print(f"  {args.latent_dim} floats x 4 bytes = {args.latent_dim * 4} bytes/frame (unquantized)")
    print(f"  -> FSQ {model.quantizer.d_fsq} dims x {model.quantizer.levels[0]} levels = {bits_per_frame} bits/frame")
    print(f"  -> 50 fps = {bits_per_frame * 50 / 1000:.1f} kbps")

    # ── Discriminator ──
    disc = MultiScaleDiscriminator().to(device)
    n_disc_params = sum(p.numel() for p in disc.parameters())
    print(f"Discriminator parameters: {n_disc_params:,}")

    # ── Loss ──
    stft_loss_fn = MultiResolutionSTFTLoss(fft_sizes=(64, 128, 256)).to(device)
    mel_loss_fn = MelSpectrogramLoss(sr=16000, n_fft=256, hop_length=64, n_mels=64).to(device)

    # GAN warmup: don't enable GAN loss until the model has basic reconstruction
    W_GAN = 0.0
    W_GAN_TARGET = 0.1  # ramp up after warmup epochs
    W_FEAT = 2.0        # feature matching weight (high — direct texture supervision)
    W_R1 = 10.0         # R1 gradient penalty weight
    W_MEL = 1.0         # mel spectrogram loss weight
    GAN_WARMUP_EPOCHS = 10

    # ── Optimizers ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    opt_disc = torch.optim.AdamW(
        disc.parameters(), lr=args.lr, weight_decay=1e-4
    )

    # ── Dataset ──
    print(f"\nLoading LibriSpeech {args.dataset}...")
    train_dataset = LibriSpeechChunks(
        root=args.data_dir, url=args.dataset, download=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    print("Loading LibriSpeech dev-clean...")
    val_dataset = LibriSpeechChunks(
        root=args.data_dir, url="dev-clean", download=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # ── Scheduler ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs
    )

    # ── EMA ──
    ema_shadow = copy.deepcopy(model.state_dict())

    # ── Resume ──
    start_epoch = 0
    best_loss = float("inf")
    ckpt_latest = output_dir / "ckpt_latest.pt"
    if ckpt_latest.exists():
        print(f"\nResuming from {ckpt_latest}")
        ckpt = torch.load(ckpt_latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["opt_state_dict"])
        if "disc_state_dict" in ckpt:
            disc.load_state_dict(ckpt["disc_state_dict"])
            opt_disc.load_state_dict(ckpt["opt_disc_state_dict"])
        ema_shadow = ckpt["ema_shadow"]
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        print(f"  Resumed at epoch {start_epoch}, best_loss={best_loss:.6f}")

    # ── Training loop ──
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print("=" * 70)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        disc.train()
        epoch_loss = 0.0
        n_steps = 0

        # GAN warmup ramp
        if epoch >= GAN_WARMUP_EPOCHS:
            W_GAN = min(W_GAN_TARGET, W_GAN_TARGET * (epoch - GAN_WARMUP_EPOCHS + 1) / 10)
        else:
            W_GAN = 0.0

        for step, chunks in enumerate(train_loader):
            chunks = chunks.to(device)  # [B, 1, 320]

            # ── Generator step ──
            recon, z_cont, z_q = model(chunks)

            # MSE loss (time domain)
            mse_loss = F.mse_loss(recon, chunks)

            # Multi-resolution STFT loss (frequency domain)
            stft_loss = stft_loss_fn(recon, chunks)

            # Quantization commitment loss
            quant_loss = F.mse_loss(z_cont, z_q.detach())

            # Mel spectrogram loss (perceptually weighted)
            mel_loss = mel_loss_fn(recon, chunks)

            # GAN generator loss + feature matching
            if W_GAN > 0:
                fake_outputs = disc(recon)
                g_adv_loss = gen_loss_fn(fake_outputs)
                feat_loss = feature_matching_loss(disc, chunks, recon)
            else:
                g_adv_loss = torch.tensor(0.0, device=device)
                feat_loss = torch.tensor(0.0, device=device)

            total_loss = (mse_loss + stft_loss + W_MEL * mel_loss
                         + 0.1 * quant_loss
                         + W_GAN * g_adv_loss
                         + W_GAN * W_FEAT * feat_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # ── Discriminator step (every other step) ──
            if W_GAN > 0 and step % 2 == 0:
                with torch.no_grad():
                    recon_d = model(chunks)[0]
                real_outputs = disc(chunks)
                fake_outputs = disc(recon_d.detach())
                d_loss = disc_loss_fn(real_outputs, fake_outputs)

                # R1 gradient penalty (every 16 steps to save compute)
                if step % 16 == 0:
                    r1_loss = r1_gradient_penalty(disc, chunks)
                    d_loss = d_loss + W_R1 * r1_loss

                opt_disc.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                opt_disc.step()

            ema_update(ema_shadow, model, decay=0.999)

            epoch_loss += total_loss.item()
            n_steps += 1

            if (step + 1) % 50 == 0:
                gan_str = f"gan={g_adv_loss.item():.4f} feat={feat_loss.item():.4f} " if W_GAN > 0 else ""
                print(
                    f"  epoch {epoch+1}/{args.num_epochs} step {step+1}/{len(train_loader)} | "
                    f"total={total_loss.item():.4f} mse={mse_loss.item():.4f} "
                    f"stft={stft_loss.item():.4f} mel={mel_loss.item():.4f} "
                    f"quant={quant_loss.item():.4f} {gan_str}"
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_steps, 1)

        # ── Validation ──
        model.eval()
        val_loss_sum = 0.0
        val_snr_sum = 0.0
        val_pesq_scores = []
        val_steps = 0

        # Load EMA weights for validation
        orig_state = copy.deepcopy(model.state_dict())
        ema_load(model, ema_shadow)

        with torch.no_grad():
            for chunks in val_loader:
                chunks = chunks.to(device)
                recon, z_cont, z_q = model(chunks)

                mse_loss = F.mse_loss(recon, chunks)
                stft_loss = stft_loss_fn(recon, chunks)
                quant_loss = F.mse_loss(z_cont, z_q.detach())
                total_loss = mse_loss + stft_loss + 0.1 * quant_loss

                val_loss_sum += total_loss.item()
                val_snr_sum += compute_snr(chunks, recon)

                # PESQ every 10 epochs (expensive, ~1 sample per batch)
                if (epoch + 1) % 10 == 0 and val_steps < 10:
                    pesq_score = compute_pesq_batch(chunks[:4], recon[:4])
                    if pesq_score is not None:
                        val_pesq_scores.append(pesq_score)

                val_steps += 1

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        avg_val_snr = val_snr_sum / max(val_steps, 1)
        avg_pesq = sum(val_pesq_scores) / len(val_pesq_scores) if val_pesq_scores else None

        # Restore training weights
        model.load_state_dict(orig_state)

        pesq_str = f" PESQ={avg_pesq:.2f}" if avg_pesq is not None else ""
        print(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"val_SNR={avg_val_snr:.2f} dB{pesq_str}"
        )

        # ── Checkpointing ──
        ckpt_data = {
            "model_state_dict": model.state_dict(),
            "opt_state_dict": optimizer.state_dict(),
            "disc_state_dict": disc.state_dict(),
            "opt_disc_state_dict": opt_disc.state_dict(),
            "ema_shadow": ema_shadow,
            "epoch": epoch,
            "best_loss": best_loss,
            "latent_dim": args.latent_dim,
        }

        torch.save(ckpt_data, output_dir / "ckpt_latest.pt")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            ckpt_data["best_loss"] = best_loss
            # Save EMA weights as best checkpoint
            ckpt_data_best = ckpt_data.copy()
            ckpt_data_best["model_state_dict"] = copy.deepcopy(ema_shadow)
            torch.save(ckpt_data_best, output_dir / "ckpt_best.pt")
            print(f"  -> New best model saved (val_loss={best_loss:.4f})")

    print("=" * 70)
    print(f"Training complete. Best val loss: {best_loss:.4f}")
    print(f"Checkpoints in: {output_dir}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SPPF Audio Autoencoder Training")
    parser.add_argument("--output_dir", type=str, default="outputs_audio",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="Latent dimension size")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root directory for LibriSpeech data")
    parser.add_argument("--dataset", type=str, default="train-clean-100",
                        help="LibriSpeech split for training (dev-clean for quick test)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
