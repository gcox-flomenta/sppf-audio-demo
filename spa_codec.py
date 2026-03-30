"""
spa_codec.py -- SPA (Semantic Phonon Audio) Codec

Binary codec for storing SPPF audio autoencoder compressed z-vectors.
Supports raw float32, GRQ 8-bit, and GRQ 4-bit compression modes.

Usage:
    # Encode audio to .spa
    python spa_codec.py encode recording.wav recording.spa --checkpoint model.pt --compression grq8

    # Decode .spa back to audio
    python spa_codec.py decode recording.spa output.wav --checkpoint model.pt

    # Inspect .spa file
    python spa_codec.py info recording.spa
"""

import argparse
import math
import struct
import sys
from dataclasses import dataclass
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


# ─────────────────────────────────────────────────────────────────────────────
# .spa Format Constants
# ─────────────────────────────────────────────────────────────────────────────

SPA_MAGIC = b"SPA\x01"
SPA_HEADER_SIZE = 32
SPA_VERSION = 1

COMPRESSION_RAW = 0
COMPRESSION_GRQ8 = 1
COMPRESSION_GRQ4 = 2

COMPRESSION_NAMES = {
    COMPRESSION_RAW: "raw",
    COMPRESSION_GRQ8: "grq8",
    COMPRESSION_GRQ4: "grq4",
}

COMPRESSION_FROM_NAME = {v: k for k, v in COMPRESSION_NAMES.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture (copied from train_audio_sppf.py -- canonical versions)
# ─────────────────────────────────────────────────────────────────────────────

class FSQQuantizer(nn.Module):
    """Finite Scalar Quantization -- codebook-free, collapse-proof."""

    def __init__(self, latent_dim: int = 512, levels: list = None):
        super().__init__()
        if levels is None:
            levels = [5] * 64
        self.levels = levels
        self.d_fsq = len(levels)
        self.latent_dim = latent_dim
        self.pre_norm = nn.LayerNorm(latent_dim)
        self.project_down = nn.Linear(latent_dim, self.d_fsq)
        self.project_up = nn.Linear(self.d_fsq, latent_dim)
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.project_down.weight, std=2.0)
        nn.init.zeros_(self.project_down.bias)
        nn.init.normal_(self.project_up.weight, std=0.02)
        nn.init.zeros_(self.project_up.bias)
        self.register_buffer("_levels_t", torch.tensor(levels, dtype=torch.float32))
        self.bits_per_frame = sum(math.ceil(math.log2(lv)) for lv in levels)

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        half = (self._levels_t - 1) / 2
        return (2.0 * torch.sigmoid(1.6 * z) - 1.0) * half

    def forward(self, z: torch.Tensor):
        z_normed = self.pre_norm(z)
        z_low = self.project_down(z_normed)
        z_bounded = self._bound(z_low)
        z_hat = torch.round(z_bounded)
        z_hat_st = z_bounded + (z_hat - z_bounded).detach()
        z_q = self.project_up(z_hat_st) * self.output_scale
        return z_q, z


class ResBlock1d(nn.Module):
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


class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial = weight_norm(nn.Conv1d(1, 32, 7, padding=3))
        self.block1 = EncoderBlock(32, 64, stride=2, kernel_size=4, padding=1)
        self.block2 = EncoderBlock(64, 128, stride=4, kernel_size=8, padding=2)
        self.block3 = EncoderBlock(128, 256, stride=5, kernel_size=5, padding=0)
        self.block4 = EncoderBlock(256, 512, stride=8, kernel_size=8, padding=0)
        self.proj = nn.Linear(512, latent_dim) if latent_dim != 512 else nn.Identity()

    def forward(self, x):
        x = self.initial(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        z = x.squeeze(-1)
        return self.proj(z)


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


class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.proj = nn.Linear(latent_dim, 512) if latent_dim != 512 else nn.Identity()
        self.block1 = DecoderBlock(512, 256, stride=8, kernel_size=8, padding=0)
        self.block2 = DecoderBlock(256, 128, stride=5, kernel_size=5, padding=0)
        self.block3 = DecoderBlock(128, 64, stride=4, kernel_size=8, padding=2)
        self.block4 = DecoderBlock(64, 32, stride=2, kernel_size=4, padding=1)
        self.final = weight_norm(nn.Conv1d(32, 1, 7, padding=3))

    def forward(self, z):
        x = self.proj(z).unsqueeze(-1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.final(x)
        return x.clamp(-1.0, 1.0)


class SPPFAudioAutoencoder(nn.Module):
    def __init__(self, latent_dim=512, use_fsq=False):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.use_fsq = use_fsq
        if use_fsq:
            self.quantizer = FSQQuantizer(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        if self.use_fsq:
            z_q, z_cont = self.quantizer(z)
            recon = self.decoder(z_q)
            return recon, z_cont, z_q
        else:
            recon = self.decoder(z)
            return recon, z, z


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O helpers (copied from infer_audio.py)
# ─────────────────────────────────────────────────────────────────────────────

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
    try:
        import subprocess, tempfile, os
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
            import wave
            import numpy as np
            with wave.open(tmp_wav) as wf:
                sr = wf.getframerate()
                data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        os.unlink(tmp_wav)
        if hasattr(data, 'ndim') and data.ndim > 1:
            data = data.mean(axis=1)
        return torch.from_numpy(data).unsqueeze(0).float(), sr
    except ImportError:
        pass
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

    print(f"ERROR: Cannot load {suffix} files. Install imageio-ffmpeg or pydub.")
    sys.exit(1)


def resample_audio(waveform, orig_sr, target_sr=16000):
    """Resample waveform to target sample rate."""
    if orig_sr == target_sr:
        return waveform
    if HAS_TORCHAUDIO:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        return resampler(waveform)
    else:
        ratio = target_sr / orig_sr
        new_length = int(waveform.shape[1] * ratio)
        return F.interpolate(
            waveform.unsqueeze(0), size=new_length, mode="linear", align_corners=False
        ).squeeze(0)


def save_wav(waveform, path: Path, sr=16000):
    """Save waveform [1, T] as WAV."""
    if HAS_SOUNDFILE:
        sf.write(str(path), waveform.squeeze(0).numpy(), sr)
    elif HAS_TORCHAUDIO:
        torchaudio.save(str(path), waveform, sr)
    else:
        print(f"WARNING: Cannot save {path} -- no audio backend available.")


# ─────────────────────────────────────────────────────────────────────────────
# GRQ (Global Range Quantization)
# ─────────────────────────────────────────────────────────────────────────────

def grq_encode(z: torch.Tensor, n_bits: int):
    """
    GRQ compression: float32 z-vectors -> quantized integers + per-frame scale.

    Args:
        z: [num_frames, latent_dim] float32 tensor
        n_bits: 8 or 4

    Returns:
        q: quantized integers (int8 for 8-bit, int8 with range -7..7 for 4-bit)
        scale: [num_frames, 1] float32 per-frame scale factors
    """
    # Per-frame scale factor: max absolute value
    scale = z.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)  # [N, 1]
    z_norm = z / scale  # normalized to [-1, 1]

    n_levels = 2 ** n_bits
    half = n_levels // 2 - 1  # 127 for 8-bit, 7 for 4-bit

    # Quantize to integer levels
    q = (z_norm * half).round().clamp(-half, half).to(torch.int8)

    return q, scale


def grq_decode(q: torch.Tensor, scale: torch.Tensor, n_bits: int):
    """
    GRQ decompression: quantized integers + scale -> float32 z-vectors.

    Args:
        q: quantized integers
        scale: [num_frames, 1] float32 per-frame scale factors
        n_bits: 8 or 4

    Returns:
        z: [num_frames, latent_dim] float32 tensor
    """
    n_levels = 2 ** n_bits
    half = n_levels // 2 - 1
    z = q.float() / half * scale
    return z


# ─────────────────────────────────────────────────────────────────────────────
# .spa Header
# ─────────────────────────────────────────────────────────────────────────────

def pack_header(sample_rate, frame_size, latent_dim, compression, num_frames, duration_ms):
    """Pack a 32-byte .spa header."""
    header = struct.pack(
        "<4sHHHHHII10s",
        SPA_MAGIC,                  # 4 bytes: magic
        SPA_VERSION,                # 2 bytes: version
        sample_rate,                # 2 bytes: sample rate
        frame_size,                 # 2 bytes: frame size in samples
        latent_dim,                 # 2 bytes: latent dim
        compression,                # 2 bytes: compression type
        num_frames,                 # 4 bytes: num frames
        duration_ms,                # 4 bytes: duration in ms
        b"\x00" * 10,              # 10 bytes: reserved
    )
    assert len(header) == SPA_HEADER_SIZE, f"Header is {len(header)} bytes, expected {SPA_HEADER_SIZE}"
    return header


def unpack_header(data: bytes):
    """Unpack a 32-byte .spa header. Returns dict."""
    if len(data) < SPA_HEADER_SIZE:
        raise ValueError(f"Header too short: {len(data)} bytes, need {SPA_HEADER_SIZE}")

    magic, version, sample_rate, frame_size, latent_dim, compression, num_frames, duration_ms, _reserved = \
        struct.unpack("<4sHHHHHII10s", data[:SPA_HEADER_SIZE])

    if magic != SPA_MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}, expected {SPA_MAGIC!r}")
    if version != SPA_VERSION:
        raise ValueError(f"Unsupported version: {version}, expected {SPA_VERSION}")

    return {
        "version": version,
        "sample_rate": sample_rate,
        "frame_size": frame_size,
        "latent_dim": latent_dim,
        "compression": compression,
        "compression_name": COMPRESSION_NAMES.get(compression, f"unknown({compression})"),
        "num_frames": num_frames,
        "duration_ms": duration_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SPAStats
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SPAStats:
    """Statistics from encoding/decoding."""
    original_size: int          # bytes
    spa_size: int               # bytes
    compression_ratio: float
    bitrate_kbps: float
    duration_seconds: float
    num_frames: int
    latent_dim: int
    compression_type: str       # "raw", "grq8", "grq4"

    def print_summary(self, label="SPA Codec"):
        print(f"\n{'=' * 60}")
        print(f"  {label} Statistics")
        print(f"{'=' * 60}")
        print(f"  Duration:          {self.duration_seconds:.2f}s")
        print(f"  Frames:            {self.num_frames}")
        print(f"  Latent dim:        {self.latent_dim}")
        print(f"  Compression:       {self.compression_type}")
        print(f"  Original size:     {self.original_size:,} bytes ({self.original_size / 1024:.1f} KB)")
        print(f"  .spa size:         {self.spa_size:,} bytes ({self.spa_size / 1024:.1f} KB)")
        print(f"  Compression ratio: {self.compression_ratio:.1f}x")
        print(f"  Bitrate:           {self.bitrate_kbps:.1f} kbps")
        print(f"  Raw PCM bitrate:   256.0 kbps (16kHz 16-bit)")
        print(f"{'=' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# Frame serialization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _frame_size_bytes(latent_dim: int, compression: int) -> int:
    """Bytes per frame for a given compression mode."""
    if compression == COMPRESSION_RAW:
        return latent_dim * 4  # float32
    elif compression == COMPRESSION_GRQ8:
        return latent_dim + 4  # uint8 per dim + float32 scale
    elif compression == COMPRESSION_GRQ4:
        return (latent_dim // 2) + 4  # packed nibbles + float32 scale
    else:
        raise ValueError(f"Unknown compression: {compression}")


def _serialize_frames_raw(z: torch.Tensor) -> bytes:
    """Serialize z-vectors as raw float32. z: [N, D]"""
    return z.contiguous().cpu().numpy().tobytes()


def _deserialize_frames_raw(data: bytes, num_frames: int, latent_dim: int) -> torch.Tensor:
    """Deserialize raw float32 frames."""
    import numpy as np
    arr = np.frombuffer(data, dtype=np.float32).reshape(num_frames, latent_dim)
    return torch.from_numpy(arr.copy())


def _serialize_frames_grq8(z: torch.Tensor) -> bytes:
    """Serialize z-vectors with GRQ 8-bit. z: [N, D]"""
    q, scale = grq_encode(z, n_bits=8)
    buf = bytearray()
    for i in range(z.shape[0]):
        # Scale as float32 (4 bytes)
        buf.extend(struct.pack("<f", scale[i, 0].item()))
        # Quantized values as int8 (latent_dim bytes)
        # Convert from int8 to unsigned for storage: val + 127 -> uint8
        q_unsigned = (q[i].to(torch.int16) + 127).clamp(0, 254).to(torch.uint8)
        buf.extend(q_unsigned.numpy().tobytes())
    return bytes(buf)


def _deserialize_frames_grq8(data: bytes, num_frames: int, latent_dim: int) -> torch.Tensor:
    """Deserialize GRQ 8-bit frames back to float32 z-vectors."""
    import numpy as np
    frame_bytes = latent_dim + 4
    z_list = []
    for i in range(num_frames):
        offset = i * frame_bytes
        scale_val = struct.unpack("<f", data[offset:offset + 4])[0]
        q_unsigned = np.frombuffer(data[offset + 4:offset + 4 + latent_dim], dtype=np.uint8)
        q_signed = torch.from_numpy(q_unsigned.astype(np.int16) - 127).to(torch.int8)
        scale = torch.tensor([[scale_val]])
        z_frame = grq_decode(q_signed.unsqueeze(0), scale, n_bits=8)
        z_list.append(z_frame)
    return torch.cat(z_list, dim=0)


def _serialize_frames_grq4(z: torch.Tensor) -> bytes:
    """Serialize z-vectors with GRQ 4-bit (packed nibbles). z: [N, D]"""
    q, scale = grq_encode(z, n_bits=4)
    latent_dim = z.shape[1]
    buf = bytearray()
    for i in range(z.shape[0]):
        # Scale as float32 (4 bytes)
        buf.extend(struct.pack("<f", scale[i, 0].item()))
        # Pack pairs of 4-bit values into bytes
        # q values are in range -7..7, shift to 0..14 for nibble storage
        q_shifted = (q[i].to(torch.int16) + 7).clamp(0, 14).to(torch.uint8)
        packed = bytearray()
        for j in range(0, latent_dim, 2):
            hi = q_shifted[j].item()
            lo = q_shifted[j + 1].item() if j + 1 < latent_dim else 0
            packed.append((hi << 4) | lo)
        buf.extend(packed)
    return bytes(buf)


def _deserialize_frames_grq4(data: bytes, num_frames: int, latent_dim: int) -> torch.Tensor:
    """Deserialize GRQ 4-bit packed frames back to float32 z-vectors."""
    packed_bytes = latent_dim // 2
    frame_bytes = packed_bytes + 4
    z_list = []
    for i in range(num_frames):
        offset = i * frame_bytes
        scale_val = struct.unpack("<f", data[offset:offset + 4])[0]
        packed = data[offset + 4:offset + 4 + packed_bytes]
        # Unpack nibbles
        q_vals = []
        for byte_val in packed:
            hi = (byte_val >> 4) & 0x0F
            lo = byte_val & 0x0F
            q_vals.append(hi)
            q_vals.append(lo)
        q_vals = q_vals[:latent_dim]  # trim if latent_dim is odd
        q_shifted = torch.tensor(q_vals, dtype=torch.int16) - 7  # back to -7..7
        q_signed = q_shifted.to(torch.int8)
        scale = torch.tensor([[scale_val]])
        z_frame = grq_decode(q_signed.unsqueeze(0), scale, n_bits=4)
        z_list.append(z_frame)
    return torch.cat(z_list, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, latent_dim: int, device: torch.device):
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

    # Auto-detect latent_dim from checkpoint if available
    ckpt_latent_dim = ckpt.get("latent_dim", latent_dim)
    if ckpt_latent_dim != latent_dim:
        print(f"  NOTE: checkpoint latent_dim={ckpt_latent_dim}, overriding --latent-dim={latent_dim}")
        latent_dim = ckpt_latent_dim

    model = SPPFAudioAutoencoder(latent_dim=latent_dim).to(device)

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
        print(f"  Weights: {weight_source}  |  latent_dim={latent_dim}")
    except RuntimeError as e:
        if weight_source == "EMA" and "model_state_dict" in ckpt:
            print(f"  EMA load failed ({e}), trying standard weights...")
            model.load_state_dict(ckpt["model_state_dict"])
            print("  Weights: standard (fallback)")
        else:
            raise

    model.eval()
    return model, latent_dim


# ─────────────────────────────────────────────────────────────────────────────
# SPAEncoder
# ─────────────────────────────────────────────────────────────────────────────

class SPAEncoder:
    """Encode audio to .spa format."""

    def __init__(self, model_path: str, latent_dim: int = 128, compression: int = COMPRESSION_GRQ8,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.compression = compression
        self.model, self.latent_dim = load_model(Path(model_path), latent_dim, self.device)
        self.sample_rate = 16000
        self.frame_size = 320  # 20ms at 16kHz

    def encode_file(self, audio_path: str, output_path: str, batch_size: int = 256) -> SPAStats:
        """Encode audio file to .spa file."""
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        if not audio_path.exists():
            print(f"ERROR: audio file not found: {audio_path}")
            sys.exit(1)

        # 1. Load audio
        print(f"Loading audio: {audio_path.name}")
        waveform, orig_sr = load_audio(audio_path)

        # 2. Resample to 16kHz mono
        waveform = resample_audio(waveform, orig_sr, self.sample_rate)

        # Normalize to [-1, 1]
        max_val = waveform.abs().max().clamp(min=1e-8)
        waveform = waveform / max_val

        total_samples = waveform.shape[1]
        duration_s = total_samples / self.sample_rate

        # 3. Split into 320-sample chunks
        n_chunks = total_samples // self.frame_size
        if n_chunks == 0:
            # Pad short audio to at least one frame
            pad_needed = self.frame_size - total_samples
            waveform = F.pad(waveform, (0, pad_needed))
            n_chunks = 1
            total_samples = self.frame_size

        waveform_trimmed = waveform[:, :n_chunks * self.frame_size]
        chunks = waveform_trimmed.view(n_chunks, 1, self.frame_size)  # [N, 1, 320]

        remainder = total_samples - n_chunks * self.frame_size
        print(f"  Duration: {duration_s:.2f}s  |  {self.sample_rate} Hz  |  {n_chunks} frames")
        if remainder > 0:
            print(f"  Trimmed {remainder} trailing samples ({remainder / self.sample_rate * 1000:.1f}ms)")

        # 4. Encode each chunk through the model's encoder
        print(f"Encoding {n_chunks} frames...")
        all_z = []
        with torch.no_grad():
            for start in range(0, n_chunks, batch_size):
                end = min(start + batch_size, n_chunks)
                batch = chunks[start:end].to(self.device)  # [B, 1, 320]
                z = self.model.encoder(batch)  # [B, latent_dim]
                all_z.append(z.cpu())

        z_all = torch.cat(all_z, dim=0)  # [N, latent_dim]

        # 5. Apply compression (GRQ) and serialize
        print(f"Compressing with {COMPRESSION_NAMES[self.compression]}...")
        if self.compression == COMPRESSION_RAW:
            frame_data = _serialize_frames_raw(z_all)
        elif self.compression == COMPRESSION_GRQ8:
            frame_data = _serialize_frames_grq8(z_all)
        elif self.compression == COMPRESSION_GRQ4:
            frame_data = _serialize_frames_grq4(z_all)
        else:
            raise ValueError(f"Unknown compression: {self.compression}")

        # 6. Write .spa file
        duration_ms = int(duration_s * 1000)
        header = pack_header(
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            latent_dim=self.latent_dim,
            compression=self.compression,
            num_frames=n_chunks,
            duration_ms=duration_ms,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(header)
            f.write(frame_data)

        spa_size = SPA_HEADER_SIZE + len(frame_data)
        original_size = n_chunks * self.frame_size * 2  # 16-bit PCM

        stats = SPAStats(
            original_size=original_size,
            spa_size=spa_size,
            compression_ratio=original_size / max(spa_size, 1),
            bitrate_kbps=spa_size * 8 / max(duration_s, 0.001) / 1000,
            duration_seconds=duration_s,
            num_frames=n_chunks,
            latent_dim=self.latent_dim,
            compression_type=COMPRESSION_NAMES[self.compression],
        )

        print(f"Wrote {output_path} ({spa_size:,} bytes)")
        return stats

    def encode_chunks(self, audio_chunks: torch.Tensor) -> bytes:
        """
        Encode raw PCM chunks to compressed bytes. For real-time streaming.

        Args:
            audio_chunks: [B, 1, 320] tensor of PCM audio frames

        Returns:
            Compressed bytes (without header -- caller manages framing)
        """
        with torch.no_grad():
            z = self.model.encoder(audio_chunks.to(self.device))  # [B, latent_dim]
            z = z.cpu()

        if self.compression == COMPRESSION_RAW:
            return _serialize_frames_raw(z)
        elif self.compression == COMPRESSION_GRQ8:
            return _serialize_frames_grq8(z)
        elif self.compression == COMPRESSION_GRQ4:
            return _serialize_frames_grq4(z)
        else:
            raise ValueError(f"Unknown compression: {self.compression}")


# ─────────────────────────────────────────────────────────────────────────────
# SPADecoder
# ─────────────────────────────────────────────────────────────────────────────

class SPADecoder:
    """Decode .spa format back to audio."""

    def __init__(self, model_path: str, latent_dim: int = 128, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, self.latent_dim = load_model(Path(model_path), latent_dim, self.device)

    def decode_file(self, spa_path: str, output_path: str, batch_size: int = 256) -> SPAStats:
        """Decode .spa file to .wav file."""
        spa_path = Path(spa_path)
        output_path = Path(output_path)

        if not spa_path.exists():
            print(f"ERROR: .spa file not found: {spa_path}")
            sys.exit(1)

        # 1. Read file
        with open(spa_path, "rb") as f:
            raw = f.read()

        # 2. Parse header
        header = unpack_header(raw[:SPA_HEADER_SIZE])
        frame_data = raw[SPA_HEADER_SIZE:]

        print(f"Decoding {spa_path.name}")
        print(f"  Version:     {header['version']}")
        print(f"  Sample rate: {header['sample_rate']} Hz")
        print(f"  Frame size:  {header['frame_size']} samples")
        print(f"  Latent dim:  {header['latent_dim']}")
        print(f"  Compression: {header['compression_name']}")
        print(f"  Frames:      {header['num_frames']}")
        print(f"  Duration:    {header['duration_ms']}ms")

        num_frames = header["num_frames"]
        latent_dim = header["latent_dim"]
        compression = header["compression"]
        sample_rate = header["sample_rate"]

        # 3. Decompress frames
        print(f"Decompressing {num_frames} frames...")
        if compression == COMPRESSION_RAW:
            z_all = _deserialize_frames_raw(frame_data, num_frames, latent_dim)
        elif compression == COMPRESSION_GRQ8:
            z_all = _deserialize_frames_grq8(frame_data, num_frames, latent_dim)
        elif compression == COMPRESSION_GRQ4:
            z_all = _deserialize_frames_grq4(frame_data, num_frames, latent_dim)
        else:
            raise ValueError(f"Unknown compression type: {compression}")

        # 4. Decode each z-vector through the model's decoder
        print(f"Decoding z-vectors to audio...")
        all_recon = []
        with torch.no_grad():
            for start in range(0, num_frames, batch_size):
                end = min(start + batch_size, num_frames)
                z_batch = z_all[start:end].to(self.device)  # [B, latent_dim]
                recon = self.model.decoder(z_batch)  # [B, 1, 320]
                all_recon.append(recon.cpu())

        # 5. Concatenate and save as .wav
        all_recon = torch.cat(all_recon, dim=0)  # [N, 1, 320]
        waveform = all_recon.squeeze(1).reshape(1, -1)  # [1, N*320]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_wav(waveform, output_path, sample_rate)

        duration_s = header["duration_ms"] / 1000.0
        spa_size = len(raw)
        original_size = num_frames * header["frame_size"] * 2  # 16-bit PCM

        stats = SPAStats(
            original_size=original_size,
            spa_size=spa_size,
            compression_ratio=original_size / max(spa_size, 1),
            bitrate_kbps=spa_size * 8 / max(duration_s, 0.001) / 1000,
            duration_seconds=duration_s,
            num_frames=num_frames,
            latent_dim=latent_dim,
            compression_type=COMPRESSION_NAMES.get(compression, f"unknown({compression})"),
        )

        print(f"Wrote {output_path} ({waveform.shape[1]} samples, {duration_s:.2f}s)")
        return stats

    def decode_chunks(self, compressed_bytes: bytes, compression: int = COMPRESSION_GRQ8,
                      latent_dim: int = None) -> torch.Tensor:
        """
        Decode compressed bytes to PCM. For real-time streaming.

        Args:
            compressed_bytes: Compressed frame bytes (without header)
            compression: Compression type used
            latent_dim: Latent dimension (defaults to self.latent_dim)

        Returns:
            [B, 1, 320] tensor of decoded PCM audio frames
        """
        if latent_dim is None:
            latent_dim = self.latent_dim

        frame_bytes = _frame_size_bytes(latent_dim, compression)
        num_frames = len(compressed_bytes) // frame_bytes

        if compression == COMPRESSION_RAW:
            z = _deserialize_frames_raw(compressed_bytes, num_frames, latent_dim)
        elif compression == COMPRESSION_GRQ8:
            z = _deserialize_frames_grq8(compressed_bytes, num_frames, latent_dim)
        elif compression == COMPRESSION_GRQ4:
            z = _deserialize_frames_grq4(compressed_bytes, num_frames, latent_dim)
        else:
            raise ValueError(f"Unknown compression: {compression}")

        with torch.no_grad():
            recon = self.model.decoder(z.to(self.device))  # [B, 1, 320]
        return recon.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Info command
# ─────────────────────────────────────────────────────────────────────────────

def print_spa_info(spa_path: str):
    """Print detailed info about a .spa file."""
    spa_path = Path(spa_path)
    if not spa_path.exists():
        print(f"ERROR: file not found: {spa_path}")
        sys.exit(1)

    file_size = spa_path.stat().st_size
    with open(spa_path, "rb") as f:
        header_bytes = f.read(SPA_HEADER_SIZE)

    header = unpack_header(header_bytes)

    latent_dim = header["latent_dim"]
    compression = header["compression"]
    num_frames = header["num_frames"]
    duration_s = header["duration_ms"] / 1000.0
    frame_bytes = _frame_size_bytes(latent_dim, compression)
    expected_data_size = SPA_HEADER_SIZE + num_frames * frame_bytes
    original_pcm_size = num_frames * header["frame_size"] * 2  # 16-bit PCM

    print(f"\n{'=' * 60}")
    print(f"  .spa File Info: {spa_path.name}")
    print(f"{'=' * 60}")
    print(f"  Magic:           {SPA_MAGIC!r}")
    print(f"  Version:         {header['version']}")
    print(f"  Sample rate:     {header['sample_rate']} Hz")
    print(f"  Frame size:      {header['frame_size']} samples ({header['frame_size'] / header['sample_rate'] * 1000:.0f}ms)")
    print(f"  Latent dim:      {latent_dim}")
    print(f"  Compression:     {header['compression_name']} (id={compression})")
    print(f"  Num frames:      {num_frames}")
    print(f"  Duration:        {duration_s:.2f}s ({header['duration_ms']}ms)")
    print(f"")
    print(f"  File size:       {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"  Header:          {SPA_HEADER_SIZE} bytes")
    print(f"  Frame data:      {file_size - SPA_HEADER_SIZE:,} bytes")
    print(f"  Bytes/frame:     {frame_bytes}")
    print(f"  Expected size:   {expected_data_size:,} bytes", end="")
    if file_size != expected_data_size:
        print(f"  ** MISMATCH (actual: {file_size}) **")
    else:
        print(f"  (OK)")
    print(f"")
    if duration_s > 0:
        bitrate = file_size * 8 / duration_s / 1000
        print(f"  .spa bitrate:    {bitrate:.1f} kbps")
    print(f"  Raw PCM size:    {original_pcm_size:,} bytes ({original_pcm_size / 1024:.1f} KB)")
    if file_size > 0:
        print(f"  Compression:     {original_pcm_size / file_size:.1f}x vs 16-bit PCM")
    print(f"  Raw PCM bitrate: 256.0 kbps (16kHz 16-bit)")
    print(f"{'=' * 60}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SPA Codec - Semantic Phonon Audio (.spa format)"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Encode
    enc = subparsers.add_parser("encode", help="Encode audio file to .spa")
    enc.add_argument("input", help="Input audio file (.wav, .mp3, .m4a, .flac)")
    enc.add_argument("output", help="Output .spa file")
    enc.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    enc.add_argument("--latent-dim", type=int, default=128, help="Latent dimension (default: 128)")
    enc.add_argument("--compression", choices=["raw", "grq8", "grq4"], default="grq8",
                     help="Compression mode (default: grq8)")
    enc.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    enc.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")

    # Decode
    dec = subparsers.add_parser("decode", help="Decode .spa file to .wav")
    dec.add_argument("input", help="Input .spa file")
    dec.add_argument("output", help="Output .wav file")
    dec.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    dec.add_argument("--latent-dim", type=int, default=128, help="Latent dimension (default: 128)")
    dec.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    dec.add_argument("--batch-size", type=int, default=256, help="Batch size for decoding")

    # Info
    info = subparsers.add_parser("info", help="Inspect .spa file header")
    info.add_argument("input", help=".spa file to inspect")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "encode":
        compression_id = COMPRESSION_FROM_NAME[args.compression]
        encoder = SPAEncoder(
            model_path=args.checkpoint,
            latent_dim=args.latent_dim,
            compression=compression_id,
            device=args.device,
        )
        stats = encoder.encode_file(args.input, args.output, batch_size=args.batch_size)
        stats.print_summary("SPA Encode")

    elif args.command == "decode":
        decoder = SPADecoder(
            model_path=args.checkpoint,
            latent_dim=args.latent_dim,
            device=args.device,
        )
        stats = decoder.decode_file(args.input, args.output, batch_size=args.batch_size)
        stats.print_summary("SPA Decode")

    elif args.command == "info":
        print_spa_info(args.input)


if __name__ == "__main__":
    main()
