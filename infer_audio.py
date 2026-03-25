"""
infer_audio.py -- SPPF Audio Autoencoder Inference Demo (self-contained)

Loads a trained SPPF audio autoencoder checkpoint, runs audio through
encode -> quantize -> decode, saves original vs reconstructed
WAV files, prints compression stats and DESE frame simulation.

Usage:
    python infer_audio.py --audio path/to/audio.wav --checkpoint outputs_audio/ckpt_best.pt
    python infer_audio.py --audio path/to/audio.wav --checkpoint outputs_audio/ckpt_best.pt --output_dir infer_output
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    import soundfile as sf
    import numpy as np
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


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
    Finite Scalar Quantization -- codebook-free, collapse-proof.

    Projects latent_dim down to d_fsq dimensions, rounds each to fixed
    levels, projects back up. Uses iFSQ activation (2*sigmoid(1.6*z)-1)
    for uniform bin utilization.

    Default: 32 dims x 5 levels = 3.7 kbps at 50 fps.
    """

    def __init__(self, latent_dim: int = 512, levels: list = None):
        super().__init__()
        if levels is None:
            levels = [5] * 64  # 64 dims, 5 levels each
        self.levels = levels
        self.d_fsq = len(levels)
        self.latent_dim = latent_dim

        # Normalize encoder output before quantization (fixes scale mismatch)
        self.pre_norm = nn.LayerNorm(latent_dim)

        # Project to/from FSQ space
        self.project_down = nn.Linear(latent_dim, self.d_fsq)
        self.project_up = nn.Linear(self.d_fsq, latent_dim)

        # Learnable output scale — initialized to match encoder output magnitude
        self.output_scale = nn.Parameter(torch.tensor(0.1))

        # Init project_down with larger std so iFSQ spans all levels
        nn.init.normal_(self.project_down.weight, std=2.0)
        nn.init.zeros_(self.project_down.bias)

        # Init project_up smaller to prevent 13x amplification
        nn.init.normal_(self.project_up.weight, std=0.02)
        nn.init.zeros_(self.project_up.bias)

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
        # Normalize encoder output to unit variance (fixes 13x scale mismatch)
        z_normed = self.pre_norm(z)

        # Project down to FSQ space
        z_low = self.project_down(z_normed)  # [B, d_fsq]

        # Bound to quantization range
        z_bounded = self._bound(z_low)

        # Round to nearest integer level
        z_hat = torch.round(z_bounded)

        # Straight-through estimator
        z_hat_st = z_bounded + (z_hat - z_bounded).detach()

        # Project back to latent_dim with learned scale
        z_q = self.project_up(z_hat_st) * self.output_scale  # [B, latent_dim]

        return z_q, z


# ─────────────────────────────────────────────
# ResBlock1d — dilated residual block with weight norm
# SoundStream-style: dilations [1, 3, 9] for exponentially growing receptive field
# NO InstanceNorm — weight_norm only
# ─────────────────────────────────────────────

class ResBlock1d(nn.Module):
    """Dilated residual block with weight norm (NO InstanceNorm).
    Dilations [1, 3, 9] give exponentially growing receptive field."""

    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(channels, channels, 1)),
        )

    def forward(self, x):
        return x + self.block(x)


# ─────────────────────────────────────────────
# EncoderBlock — 3 dilated ResBlocks + strided downsampling conv
# ─────────────────────────────────────────────

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, kernel_size=None, padding=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = 2 * stride
        if padding is None:
            padding = stride // 2
        self.res_blocks = nn.Sequential(
            ResBlock1d(in_ch, dilation=1),
            ResBlock1d(in_ch, dilation=3),
            ResBlock1d(in_ch, dilation=9),
        )
        self.downsample = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.downsample(x)
        return x


# ─────────────────────────────────────────────
# Encoder
# Input: [B, 1, 320] -> Output: [B, 512]
# Total stride: 2 * 4 * 5 * 8 = 320 (maps 320 samples to 1 vector)
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # Initial conv: [B, 1, 320] -> [B, 32, 320]
        self.initial = weight_norm(nn.Conv1d(1, 32, 7, padding=3))

        # Downsampling blocks
        # Conv1d output: (L_in + 2*pad - kernel) / stride + 1
        self.block1 = EncoderBlock(32, 64, stride=2, kernel_size=4, padding=1)     # (320+2-4)/2+1 = 160
        self.block2 = EncoderBlock(64, 128, stride=4, kernel_size=8, padding=2)    # (160+4-8)/4+1 = 40
        self.block3 = EncoderBlock(128, 256, stride=5, kernel_size=5, padding=0)   # (40+0-5)/5+1 = 8
        self.block4 = EncoderBlock(256, 512, stride=8, kernel_size=8, padding=0)   # (8+0-8)/8+1 = 1

    def forward(self, x):
        x = self.initial(x)     # [B, 32, 320]
        x = self.block1(x)      # [B, 64, 160]
        x = self.block2(x)      # [B, 128, 40]
        x = self.block3(x)      # [B, 256, 8]
        x = self.block4(x)      # [B, 512, 1]
        return x.squeeze(-1)    # [B, 512]


# ─────────────────────────────────────────────
# DecoderBlock — transposed conv upsampling + 3 dilated ResBlocks
# ─────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, kernel_size=None, padding=0, output_padding=0):
        super().__init__()
        if kernel_size is None:
            kernel_size = 2 * stride
        self.upsample = weight_norm(
            nn.ConvTranspose1d(
                in_ch, out_ch, kernel_size, stride=stride,
                padding=padding, output_padding=output_padding
            )
        )
        self.res_blocks = nn.Sequential(
            ResBlock1d(out_ch, dilation=1),
            ResBlock1d(out_ch, dilation=3),
            ResBlock1d(out_ch, dilation=9),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.res_blocks(x)
        return x


# ─────────────────────────────────────────────
# Decoder
# Input: [B, 512] -> Output: [B, 1, 320]
# Mirror of encoder with ConvTranspose1d upsampling
# ─────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # Upsampling blocks (mirror of encoder)
        # ConvTranspose1d output: (L_in - 1) * stride - 2*padding + kernel + output_padding
        # Block 1: stride 8, [B, 512, 1] -> [B, 256, 8]
        self.block1 = DecoderBlock(512, 256, stride=8, kernel_size=8, padding=0)  # (1-1)*8-0+8=8
        # Block 2: stride 5, [B, 256, 8] -> [B, 128, 40]
        self.block2 = DecoderBlock(256, 128, stride=5, kernel_size=5, padding=0)  # (8-1)*5+5=40
        # Block 3: stride 4, [B, 128, 40] -> [B, 64, 160]
        self.block3 = DecoderBlock(128, 64, stride=4, kernel_size=8, padding=2)   # (40-1)*4-4+8=160
        # Block 4: stride 2, [B, 64, 160] -> [B, 32, 320]
        self.block4 = DecoderBlock(64, 32, stride=2, kernel_size=4, padding=1)    # (160-1)*2-2+4=320

        # Final conv: [B, 32, 320] -> [B, 1, 320] -- NO Tanh, linear output
        self.final = weight_norm(nn.Conv1d(32, 1, 7, padding=3))

    def forward(self, z):
        # z: [B, 512] -> unsqueeze to [B, 512, 1]
        x = z.unsqueeze(-1)     # [B, 512, 1]
        x = self.block1(x)      # [B, 256, 8]
        x = self.block2(x)      # [B, 128, 40]
        x = self.block3(x)      # [B, 64, 160]
        x = self.block4(x)      # [B, 32, 320]
        x = self.final(x)       # [B, 1, 320]
        return x.clamp(-1.0, 1.0)  # bound output to valid audio range


# ─────────────────────────────────────────────
# Full Autoencoder
# Encoder -> FSQ Quantizer -> Decoder (no SPPFCore)
# ─────────────────────────────────────────────

class SPPFAudioAutoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.quantizer = FSQQuantizer(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)             # [B, 512]
        z_q, z_cont = self.quantizer(z) # quantized, continuous
        recon = self.decoder(z_q)       # [B, 1, 320]
        return recon, z_cont, z_q


# ---------------------------------------------
# Audio I/O helpers
# ---------------------------------------------

def load_audio(audio_path: Path):
    """Load audio file, return (waveform [1, T], sample_rate)."""
    suffix = audio_path.suffix.lower()

    # Try soundfile first (reliable for wav/flac, no torchcodec needed)
    if HAS_SOUNDFILE and suffix in {".wav", ".flac", ".ogg"}:
        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return torch.from_numpy(data).unsqueeze(0), sr

    # For non-wav/flac formats (m4a, mp3, aac, etc.), convert to wav via ffmpeg
    # Uses imageio-ffmpeg's bundled binary — no system install needed
    try:
        import subprocess, tempfile
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        tmp_wav = tempfile.mktemp(suffix=".wav")
        result = subprocess.run(
            [ffmpeg_exe, "-y", "-i", str(audio_path), "-ac", "1", "-ar", "16000", "-f", "wav", tmp_wav],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[:200]}")
        if HAS_SOUNDFILE:
            data, sr = sf.read(tmp_wav, dtype="float32")
        else:
            import wave, numpy as np
            with wave.open(tmp_wav) as wf:
                sr = wf.getframerate()
                data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        import os; os.unlink(tmp_wav)
        if hasattr(data, 'ndim') and data.ndim > 1:
            data = data.mean(axis=1)
        return torch.from_numpy(data).unsqueeze(0).float(), sr
    except ImportError:
        print("Install imageio-ffmpeg for m4a/mp3 support: pip install imageio-ffmpeg")
    except Exception as e:
        print(f"ffmpeg conversion failed: {e}")

    # Try torchaudio as last resort
    if HAS_TORCHAUDIO:
        try:
            waveform, sr = torchaudio.load(str(audio_path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform, sr
        except Exception as e:
            print(f"torchaudio failed: {e}")

    # Try soundfile for any remaining format
    if HAS_SOUNDFILE:
        try:
            data, sr = sf.read(str(audio_path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            return torch.from_numpy(data).unsqueeze(0), sr
        except Exception:
            pass

    print(f"ERROR: Cannot load {suffix} files. Install pydub (`pip install pydub`) for m4a/mp3 support.")
    sys.exit(1)


def resample_audio(waveform, orig_sr, target_sr=16000):
    """Resample waveform to target sample rate."""
    if orig_sr == target_sr:
        return waveform
    if HAS_TORCHAUDIO:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        return resampler(waveform)
    else:
        # Simple linear interpolation fallback
        ratio = target_sr / orig_sr
        new_length = int(waveform.shape[1] * ratio)
        return F.interpolate(
            waveform.unsqueeze(0), size=new_length, mode="linear", align_corners=False
        ).squeeze(0)


def save_wav(waveform, path: Path, sr=16000):
    """Save waveform [1, T] as WAV."""
    # Use soundfile first (doesn't need torchcodec)
    if HAS_SOUNDFILE:
        sf.write(str(path), waveform.squeeze(0).numpy(), sr)
    elif HAS_TORCHAUDIO:
        torchaudio.save(str(path), waveform, sr)
    else:
        print(f"WARNING: Cannot save {path} -- no audio backend available.")


# ---------------------------------------------
# Metrics
# ---------------------------------------------

def compute_snr(original, reconstructed):
    """Compute signal-to-noise ratio in dB."""
    signal_power = (original ** 2).mean()
    noise_power = ((original - reconstructed) ** 2).mean()
    if noise_power < 1e-12:
        return float('inf')
    return 10 * math.log10(signal_power.item() / noise_power.item())


def compute_mse(original, reconstructed):
    """Compute mean squared error."""
    return F.mse_loss(original, reconstructed).item()


# ---------------------------------------------
# DESE frame classification
# ---------------------------------------------

# Frame type IDs (from CLAUDE.md)
SILENCE = 0
ONSET = 1
CHANGE = 2
STEADY = 3
ANCHOR = 4

FRAME_NAMES = {
    SILENCE: "SILENCE",
    ONSET:   "ONSET",
    CHANGE:  "CHANGE",
    STEADY:  "STEADY",
    ANCHOR:  "ANCHOR",
}

# SPPF priority levels
PRIORITY = {
    ONSET:   "P0 CRITICAL",
    CHANGE:  "P1 HIGH",
    ANCHOR:  "P2 NORMAL",
    STEADY:  "P3 LOW",
    SILENCE: "P3 LOW",
}

# Whether a frame type is transmitted
TRANSMIT = {
    SILENCE: False,
    ONSET:   True,
    CHANGE:  True,
    STEADY:  False,
    ANCHOR:  True,
}


def classify_dese_frames(
    latent_vectors,         # list of [latent_dim] tensors
    chunks_rms,             # list of float RMS values per chunk
    silence_rms=0.01,       # RMS threshold for silence
    change_threshold=0.5,   # L2 distance threshold for CHANGE vs STEADY
    anchor_interval=100,    # anchor every N frames (~2s at 50fps)
):
    """
    Classify each frame as SILENCE/ONSET/CHANGE/STEADY/ANCHOR.
    Returns list of frame type IDs.
    """
    n = len(latent_vectors)
    frame_types = []
    prev_was_silence = True
    baseline = None

    for i in range(n):
        rms = chunks_rms[i]

        # Every anchor_interval frames, force an ANCHOR
        if i > 0 and i % anchor_interval == 0 and rms >= silence_rms:
            frame_types.append(ANCHOR)
            baseline = latent_vectors[i]
            continue

        # Silence detection
        if rms < silence_rms:
            frame_types.append(SILENCE)
            prev_was_silence = True
            continue

        # Onset: transition from silence to activity
        if prev_was_silence:
            frame_types.append(ONSET)
            baseline = latent_vectors[i]
            prev_was_silence = False
            continue

        prev_was_silence = False

        # Compare against baseline
        if baseline is not None:
            dist = torch.norm(latent_vectors[i] - baseline, p=2).item()
        else:
            dist = float('inf')

        if dist > change_threshold:
            frame_types.append(CHANGE)
            baseline = latent_vectors[i]
        else:
            frame_types.append(STEADY)

    return frame_types


# ---------------------------------------------
# Checkpoint loading
# ---------------------------------------------

def load_model(ckpt_path: Path, device):
    """Load model from checkpoint. Tries EMA weights first."""
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    size_mb = ckpt_path.stat().st_size / 1024 / 1024
    print(f"Loading {ckpt_path.name}  ({size_mb:.1f} MB)...")

    try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location=device)
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        sys.exit(1)

    # Auto-detect latent_dim from checkpoint
    latent_dim = ckpt.get("latent_dim", 512)
    print(f"  Architecture: SPPFAudioAutoencoder (v3)  |  latent_dim={latent_dim}")

    model = SPPFAudioAutoencoder(latent_dim=latent_dim).to(device)

    # Try EMA weights first, then standard model weights
    state_dict = None
    weight_source = None

    if "ema_shadow" in ckpt:
        state_dict = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}
        weight_source = "EMA"
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        weight_source = "standard"
    else:
        print("ERROR: checkpoint contains neither 'ema_shadow' nor 'model_state_dict'")
        sys.exit(1)

    try:
        model.load_state_dict(state_dict)
        print(f"  Weights: {weight_source}")
    except RuntimeError as e:
        if weight_source == "EMA" and "model_state_dict" in ckpt:
            print(f"  EMA load failed ({e}), trying standard weights...")
            model.load_state_dict(ckpt["model_state_dict"])
            print("  Weights: standard (fallback)")
        else:
            raise

    model.eval()
    return model, latent_dim


# ---------------------------------------------
# Main inference pipeline
# ---------------------------------------------

def run_inference(args):
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}\n")

    # -- Load model --
    model, latent_dim = load_model(Path(args.checkpoint), device)

    # -- Load audio --
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: audio file not found: {audio_path}")
        sys.exit(1)

    print(f"\nLoading audio: {audio_path.name}")
    waveform, orig_sr = load_audio(audio_path)

    # Resample to 16kHz
    target_sr = 16000
    waveform = resample_audio(waveform, orig_sr, target_sr)

    # Normalize to [-1, 1]
    max_val = waveform.abs().max().clamp(min=1e-8)
    waveform = waveform / max_val

    total_samples = waveform.shape[1]
    duration = total_samples / target_sr
    print(f"  Duration: {duration:.2f}s  |  {target_sr} Hz  |  mono")

    # -- Split into 20ms chunks (320 samples) --
    chunk_size = 320
    n_chunks = total_samples // chunk_size
    remainder = total_samples % chunk_size

    # Trim to exact multiple of chunk_size
    waveform_trimmed = waveform[:, :n_chunks * chunk_size]
    chunks = waveform_trimmed.view(1, n_chunks, chunk_size)  # [1, N, 320]
    chunks = chunks.squeeze(0)  # [N, 320]

    print(f"  Chunks: {n_chunks} x {chunk_size} samples (20ms each)")
    if remainder > 0:
        print(f"  Trimmed {remainder} trailing samples ({remainder / target_sr * 1000:.1f}ms)")

    # -- Process chunks through model --
    print(f"\nRunning inference...")
    batch_size = min(args.batch_size, n_chunks)
    all_recon = []
    all_latents = []
    chunks_rms = []

    with torch.no_grad():
        for start in range(0, n_chunks, batch_size):
            end = min(start + batch_size, n_chunks)
            batch = chunks[start:end].unsqueeze(1).to(device)  # [B, 1, 320]

            recon, z_cont, z_q = model(batch)

            all_recon.append(recon.cpu())

            # Store per-chunk latent vectors for DESE simulation
            for i in range(batch.shape[0]):
                all_latents.append(z_q[i].cpu())
                rms = batch[i].pow(2).mean().sqrt().item()
                chunks_rms.append(rms)

    # Concatenate reconstructions
    all_recon = torch.cat(all_recon, dim=0)  # [N, 1, 320]
    recon_waveform = all_recon.squeeze(1).reshape(1, -1)  # [1, N*320]
    original_waveform = waveform_trimmed

    # -- Compute quality metrics --
    snr = compute_snr(original_waveform, recon_waveform)
    mse = compute_mse(original_waveform, recon_waveform)

    # Per-chunk SNR stats
    chunk_snrs = []
    for i in range(n_chunks):
        orig_chunk = original_waveform[0, i * chunk_size:(i + 1) * chunk_size]
        recon_chunk = recon_waveform[0, i * chunk_size:(i + 1) * chunk_size]
        if orig_chunk.pow(2).mean() > 1e-10:
            chunk_snrs.append(compute_snr(orig_chunk, recon_chunk))

    # -- Save output files --
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = audio_path.stem
    orig_out = out_dir / f"{stem}_original.wav"
    recon_out = out_dir / f"{stem}_reconstructed.wav"

    save_wav(original_waveform, orig_out, target_sr)
    save_wav(recon_waveform, recon_out, target_sr)

    # -- Compression stats --
    raw_bytes_per_chunk = chunk_size * 2  # 16-bit PCM = 2 bytes/sample
    compressed_bytes_per_chunk = latent_dim  # GRQ 8-bit = 1 byte per latent dim
    raw_bitrate = target_sr * 16 / 1000  # kbps (16-bit PCM)
    compressed_bitrate = compressed_bytes_per_chunk * 50 * 8 / 1000  # 50 fps, 8 bits/byte
    compression_ratio = raw_bytes_per_chunk / compressed_bytes_per_chunk

    print(f"\n{'=' * 65}")
    print(f"SPPF Audio Autoencoder — Inference Results")
    print(f"{'=' * 65}")
    print(f"Audio: {audio_path.name} ({duration:.1f}s, {target_sr} Hz, mono)")
    print(f"Chunks: {n_chunks} x {chunk_size} samples (20ms each)")
    print(f"Compression: {raw_bytes_per_chunk} bytes/chunk -> "
          f"{compressed_bytes_per_chunk} bytes/chunk "
          f"(GRQ 8-bit) = {compression_ratio:.0f}x reduction")
    print(f"Bitrate: {raw_bitrate:.0f} kbps -> {compressed_bitrate:.1f} kbps")
    print(f"SNR: {snr:.1f} dB  (median per-chunk: "
          f"{sorted(chunk_snrs)[len(chunk_snrs)//2]:.1f} dB)" if chunk_snrs else f"SNR: {snr:.1f} dB")
    print(f"MSE: {mse:.6f}")
    print(f"\nSaved:")
    print(f"  Original:      {orig_out}")
    print(f"  Reconstructed: {recon_out}")

    # -- DESE frame simulation --
    print(f"\n{'=' * 65}")
    print(f"DESE Frame Simulation")
    print(f"{'=' * 65}")

    frame_types = classify_dese_frames(
        all_latents, chunks_rms,
        silence_rms=args.silence_rms,
        change_threshold=args.change_threshold,
        anchor_interval=args.anchor_interval,
    )

    # Count distribution
    counts = {SILENCE: 0, ONSET: 0, CHANGE: 0, STEADY: 0, ANCHOR: 0}
    for ft in frame_types:
        counts[ft] += 1

    total_dese_bytes = 0
    print(f"\n  Frame Distribution:")
    for ft in [SILENCE, STEADY, CHANGE, ONSET, ANCHOR]:
        pct = counts[ft] / n_chunks * 100 if n_chunks > 0 else 0
        if TRANSMIT[ft]:
            bytes_each = compressed_bytes_per_chunk
            total_dese_bytes += counts[ft] * bytes_each
            print(f"    {FRAME_NAMES[ft]:>8s}: {pct:5.1f}%  ({counts[ft]:4d} frames, "
                  f"{bytes_each} bytes each)  [{PRIORITY[ft]}]")
        else:
            print(f"    {FRAME_NAMES[ft]:>8s}: {pct:5.1f}%  ({counts[ft]:4d} frames, "
                  f"0 bytes)          [{PRIORITY[ft]}]")

    # Effective bitrate
    duration_trimmed = n_chunks * chunk_size / target_sr
    if duration_trimmed > 0:
        effective_bitrate = total_dese_bytes * 8 / duration_trimmed / 1000
    else:
        effective_bitrate = 0

    transmitted = sum(counts[ft] for ft in [ONSET, CHANGE, ANCHOR])
    suppressed = sum(counts[ft] for ft in [SILENCE, STEADY])
    suppression_pct = suppressed / n_chunks * 100 if n_chunks > 0 else 0

    print(f"\n  Transmitted: {transmitted} frames  |  Suppressed: {suppressed} frames ({suppression_pct:.0f}%)")
    print(f"  Effective bitrate: {effective_bitrate:.1f} kbps "
          f"(vs {compressed_bitrate:.1f} kbps base, vs {raw_bitrate:.0f} kbps raw)")
    print(f"  Total DESE bandwidth: {total_dese_bytes:,} bytes for {duration_trimmed:.1f}s of audio")

    print(f"\n{'=' * 65}")
    print(f"Done.")


# ---------------------------------------------
# CLI
# ---------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="SPPF Audio Autoencoder Inference Demo"
    )
    ap.add_argument("--audio", required=True,
                    help="Path to input audio file (.wav, .flac, .mp3)")
    ap.add_argument("--checkpoint", default="outputs_audio/ckpt_best.pt",
                    help="Path to model checkpoint")
    ap.add_argument("--output_dir", default="infer_output",
                    help="Directory for output files")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU inference")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="Batch size for chunk processing")

    # DESE simulation parameters
    ap.add_argument("--silence_rms", type=float, default=0.01,
                    help="RMS threshold for silence detection")
    ap.add_argument("--change_threshold", type=float, default=0.5,
                    help="L2 distance threshold for CHANGE vs STEADY classification")
    ap.add_argument("--anchor_interval", type=int, default=100,
                    help="Anchor frame interval (frames, ~2s at 50fps)")

    args = ap.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
