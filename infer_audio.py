"""
infer_audio.py -- SPPF Audio Autoencoder Inference Demo (self-contained)

Loads a trained SPPF audio autoencoder checkpoint, runs audio through
encode -> SPPF core -> quantize -> decode, saves original vs reconstructed
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


# ---------------------------------------------
# Golden Ratio Quantizer (inline from training)
# ---------------------------------------------

PHI = (1.0 + math.sqrt(5.0)) / 2.0


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
    """8-bit quantizer with phi-spaced bucket boundaries. STE for training."""

    def __init__(self, n_bits: int = 8):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits

        boundaries = _golden_ratio_boundaries(n_bits, max_scale=1.0)
        self.register_buffer("boundaries", boundaries)

        padded = torch.cat([
            boundaries[:1] - (boundaries[1] - boundaries[0]),
            boundaries,
            boundaries[-1:] + (boundaries[-1] - boundaries[-2]),
        ])
        rep_values = 0.5 * (padded[:-1] + padded[1:])
        self.register_buffer("rep_values", rep_values)

    def forward(self, z: torch.Tensor):
        scale = z.abs().amax().clamp(min=1e-8)
        z_scaled = z / scale

        indices = torch.bucketize(z_scaled.contiguous(), self.boundaries)
        z_q_scaled = self.rep_values[indices.long()]

        z_q_raw = z_q_scaled * scale
        z_q = z + (z_q_raw - z).detach()
        return z_q, z


# ---------------------------------------------
# ResBlock1d
# ---------------------------------------------

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


# ---------------------------------------------
# Encoder1d: [B, 1, 320] -> [B, latent_dim]
# ---------------------------------------------

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
        self.fc = nn.Linear(64 * 20, latent_dim)

    def forward(self, x):
        x = self.stage1(x)   # [B, 32, 160]
        x = self.stage2(x)   # [B, 64, 80]
        x = self.stage3(x)   # [B, 128, 40]
        x = self.stage4(x)   # [B, 64, 20]
        return self.fc(x.flatten(1))  # [B, latent_dim]


# ---------------------------------------------
# SPPFCore1d — MLP
# ---------------------------------------------

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


# ---------------------------------------------
# Decoder1d: [B, latent_dim] -> [B, 1, 320]
# ---------------------------------------------

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


# ---------------------------------------------
# Full Autoencoder
# ---------------------------------------------

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

    # Try pydub for m4a/mp3/aac (needs ffmpeg or just pydub)
    try:
        from pydub import AudioSegment
        import numpy as np
        seg = AudioSegment.from_file(str(audio_path))
        seg = seg.set_channels(1)
        sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples = samples / (2 ** (seg.sample_width * 8 - 1))  # normalize to [-1, 1]
        return torch.from_numpy(samples).unsqueeze(0), sr
    except Exception:
        pass

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
    if HAS_TORCHAUDIO:
        torchaudio.save(str(path), waveform, sr)
    elif HAS_SOUNDFILE:
        sf.write(str(path), waveform.squeeze(0).numpy(), sr)
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
    latent_dim = ckpt.get("latent_dim", 64)
    print(f"  Architecture: SPPFAudioAutoencoder  |  latent_dim={latent_dim}")

    model = SPPFAudioAutoencoder(latent_dim=latent_dim).to(device)

    try:
        if "ema_shadow" in ckpt:
            model.load_state_dict(
                {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}
            )
            print("  Weights: EMA")
        else:
            model.load_state_dict(ckpt["model_state_dict"])
            print("  Weights: standard")
    except Exception as e:
        print(f"  EMA load failed ({e}), using standard weights")
        model.load_state_dict(ckpt["model_state_dict"])

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
