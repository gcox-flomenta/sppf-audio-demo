"""
train_audio_sppf.py — SPPF Audio Autoencoder Training

Compresses 20ms speech frames (320 samples @ 16kHz) into a 64-byte latent
using Golden Ratio Quantization, then reconstructs. Replaces Opus for voice calls.

Architecture:
  Encoder (Conv1d):  [B, 1, 320] → 4 strided stages → [B, latent_dim]
  SPPF Core (MLP):   [B, latent_dim] → [B, latent_dim]
  GR Quantizer:      64 floats → 64 bytes (8-bit, phi-spaced)
  Decoder (Conv1d):  [B, latent_dim] → 4 upsample stages → [B, 1, 320]

Loss:
  MSE (time domain) + Multi-resolution STFT (frequency domain) + Quantization commitment

Dataset: LibriSpeech train-clean-100 / dev-clean (auto-download via torchaudio)

Usage:
  python train_audio_sppf.py --output_dir outputs_audio --num_epochs 50 --batch_size 64
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
# Golden Ratio Quantizer (inline)
# φ-spaced bucket boundaries for 8-bit quantization.
# Straight-through estimator: forward uses quantized, backward uses continuous.
# ─────────────────────────────────────────────

PHI = (1.0 + math.sqrt(5.0)) / 2.0  # 1.6180339887...


def _golden_ratio_boundaries(n_bits: int = 8, max_scale: float = 1.0) -> torch.Tensor:
    """Generate symmetric quantization boundaries with golden ratio width spacing."""
    n_levels = 2 ** n_bits
    n_positive = n_levels // 2 - 1

    width_ratio = PHI ** (2.0 / max(n_positive, 1))

    k = torch.arange(n_positive, dtype=torch.float64)
    widths = torch.pow(torch.tensor(width_ratio), k)

    positive = widths.cumsum(0)
    positive = positive * (max_scale / positive[-1])

    negative = -positive.flip(0)
    boundaries = torch.cat([negative, torch.zeros(1, dtype=torch.float64), positive])
    return boundaries.float()


class GoldenRatioQuantizer(nn.Module):
    """
    8-bit quantizer with phi-spaced bucket boundaries.
    Straight-through estimator for end-to-end training.
    """

    def __init__(self, n_bits: int = 8):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits

        boundaries = _golden_ratio_boundaries(n_bits, max_scale=1.0)
        self.register_buffer("boundaries", boundaries)

        # Representative values: midpoints between consecutive boundaries
        padded = torch.cat([
            boundaries[:1] - (boundaries[1] - boundaries[0]),
            boundaries,
            boundaries[-1:] + (boundaries[-1] - boundaries[-2]),
        ])
        rep_values = 0.5 * (padded[:-1] + padded[1:])
        self.register_buffer("rep_values", rep_values)

    def forward(self, z: torch.Tensor):
        """
        Quantize with straight-through estimator.

        Returns:
            z_q: quantized tensor (STE: forward=quantized, backward=continuous)
            z: original continuous tensor (for commitment loss)
        """
        # Per-tensor scaling to [-1, 1]
        scale = z.abs().amax().clamp(min=1e-8)
        z_scaled = z / scale

        # Quantize
        indices = torch.bucketize(z_scaled.contiguous(), self.boundaries)
        z_q_scaled = self.rep_values[indices.long()]

        # Unscale
        z_q_raw = z_q_scaled * scale

        # Straight-through estimator: z_q has quantized values but z's gradients
        z_q = z + (z_q_raw - z).detach()

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
        self.quantizer = GoldenRatioQuantizer(n_bits=8)
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
    print(f"\nCompression stats:")
    print(f"  {args.latent_dim} floats x 4 bytes = {args.latent_dim * 4} bytes/frame")
    print(f"  -> GRQ 8-bit: {args.latent_dim} bytes/frame")
    print(f"  -> 50 fps = {args.latent_dim * 50 * 8 / 1000:.1f} kbps")

    # ── Loss ──
    stft_loss_fn = MultiResolutionSTFTLoss(fft_sizes=(64, 128, 256)).to(device)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
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
        ema_shadow = ckpt["ema_shadow"]
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        print(f"  Resumed at epoch {start_epoch}, best_loss={best_loss:.6f}")

    # ── Training loop ──
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print("=" * 70)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        for step, chunks in enumerate(train_loader):
            chunks = chunks.to(device)  # [B, 1, 320]

            recon, z_cont, z_q = model(chunks)

            # MSE loss (time domain)
            mse_loss = F.mse_loss(recon, chunks)

            # Multi-resolution STFT loss (frequency domain)
            stft_loss = stft_loss_fn(recon, chunks)

            # Quantization commitment loss
            quant_loss = F.mse_loss(z_cont, z_q.detach())

            total_loss = mse_loss + stft_loss + 0.1 * quant_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ema_update(ema_shadow, model, decay=0.999)

            epoch_loss += total_loss.item()
            n_steps += 1

            if (step + 1) % 50 == 0:
                print(
                    f"  epoch {epoch+1}/{args.num_epochs} step {step+1}/{len(train_loader)} | "
                    f"total={total_loss.item():.4f} mse={mse_loss.item():.4f} "
                    f"stft={stft_loss.item():.4f} quant={quant_loss.item():.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_steps, 1)

        # ── Validation ──
        model.eval()
        val_loss_sum = 0.0
        val_snr_sum = 0.0
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
                val_steps += 1

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        avg_val_snr = val_snr_sum / max(val_steps, 1)

        # Restore training weights
        model.load_state_dict(orig_state)

        print(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"val_SNR={avg_val_snr:.2f} dB"
        )

        # ── Checkpointing ──
        ckpt_data = {
            "model_state_dict": model.state_dict(),
            "opt_state_dict": optimizer.state_dict(),
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
    parser.add_argument("--num_epochs", type=int, default=50,
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
