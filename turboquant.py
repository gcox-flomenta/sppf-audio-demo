"""
turboquant.py -- TurboQuant for SPPF Audio Latent Compression

Implements Google's TurboQuant (ICLR 2026, arXiv 2504.19874) for compressing
128-dim audio latent vectors. Near-optimal distortion at extreme compression.

Key idea: randomly rotate vectors to decorrelate dimensions, then apply
optimal Lloyd-Max scalar quantization per coordinate. No training needed.

Usage:
    tq = TurboQuant(dim=128, n_bits=3, seed=42)

    # Encode (128 floats -> 48 bytes at 3-bit)
    indices, scale = tq.encode(z_vector)

    # Decode (48 bytes -> 128 floats)
    z_hat = tq.decode(indices, scale)
"""

import math
import torch
import torch.nn.functional as F
import numpy as np


class TurboQuant:
    """TurboQuant: Near-optimal vector quantization via random rotation.

    1. Normalize input vector and save scale
    2. Rotate by random orthogonal matrix (decorrelates dimensions)
    3. Lloyd-Max quantize each rotated coordinate to b bits
    4. To decode: dequantize, rotate back, rescale

    The random rotation makes coordinates follow a concentrated Beta
    distribution, so per-coordinate Lloyd-Max quantization is near-optimal.
    """

    def __init__(self, dim=128, n_bits=3, seed=42):
        self.dim = dim
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits

        # Generate random orthogonal matrix via QR decomposition
        # Same matrix used for all vectors (shared between sender/receiver)
        rng = np.random.RandomState(seed)
        gaussian = rng.randn(dim, dim).astype(np.float32)
        Q, R = np.linalg.qr(gaussian)
        # Ensure proper rotation (det = +1)
        Q = Q * np.sign(np.diag(R))
        self.rotation = torch.from_numpy(Q)       # [dim, dim]
        self.rotation_t = self.rotation.T.contiguous()  # [dim, dim] for decode

        # Precompute Lloyd-Max centroids for Beta distribution
        # In high dimensions, rotated coordinates ~ N(0, 1/dim)
        # Normalized to unit norm, they follow Beta((d-1)/2, (d-1)/2) scaled to [-1,1]
        # For practical purposes, compute Lloyd-Max on N(0,1) then scale
        self.centroids, self.boundaries = self._compute_lloyd_max(n_bits)

    def _compute_lloyd_max(self, n_bits, n_iter=100):
        """Compute Lloyd-Max optimal quantizer for approximately Gaussian data.

        Lloyd-Max places quantization levels where data density is highest,
        unlike uniform quantization which wastes levels in low-density tails.
        """
        n_levels = 2 ** n_bits

        # Initialize centroids uniformly in [-3, 3] (covers >99.7% of N(0,1))
        centroids = torch.linspace(-3, 3, n_levels)

        # Lloyd-Max iteration (1D k-means with Gaussian density)
        # Generate representative samples from N(0,1)
        samples = torch.randn(100000)

        for _ in range(n_iter):
            # Compute boundaries (midpoints between consecutive centroids)
            boundaries = (centroids[:-1] + centroids[1:]) / 2

            # Assign samples to nearest centroid
            # boundaries define the partition
            full_boundaries = torch.cat([
                torch.tensor([-float('inf')]),
                boundaries,
                torch.tensor([float('inf')])
            ])

            # Update centroids as conditional means
            new_centroids = torch.zeros_like(centroids)
            for i in range(n_levels):
                mask = (samples >= full_boundaries[i]) & (samples < full_boundaries[i + 1])
                if mask.sum() > 0:
                    new_centroids[i] = samples[mask].mean()
                else:
                    new_centroids[i] = centroids[i]

            centroids = new_centroids

        # Final boundaries
        boundaries = (centroids[:-1] + centroids[1:]) / 2

        return centroids, boundaries

    def encode(self, z):
        """Encode a latent vector or batch of vectors.

        Args:
            z: [dim] or [B, dim] float tensor

        Returns:
            indices: [dim] or [B, dim] uint8 tensor (quantized)
            scale: [] or [B] float tensor (per-vector norm)
        """
        single = z.dim() == 1
        if single:
            z = z.unsqueeze(0)

        B = z.shape[0]
        device = z.device

        # Save scale (norm) and normalize to unit sphere
        scale = z.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 1]
        z_norm = z / scale  # [B, dim], unit norm

        # Rotate to decorrelate dimensions
        rot = self.rotation.to(device)
        z_rot = z_norm @ rot.T  # [B, dim]

        # Scale rotated coordinates to match the Lloyd-Max quantizer range
        # After rotation of unit vector, coordinates have std ≈ 1/sqrt(dim)
        coord_scale = math.sqrt(self.dim)
        z_scaled = z_rot * coord_scale  # now approximately N(0, 1) per coordinate

        # Lloyd-Max quantization: find nearest centroid for each coordinate
        boundaries = self.boundaries.to(device)
        indices = torch.bucketize(z_scaled, boundaries).to(torch.uint8)  # [B, dim]

        if single:
            return indices.squeeze(0), scale.squeeze()
        return indices, scale.squeeze(-1)

    def decode(self, indices, scale):
        """Decode quantized indices back to latent vector.

        Args:
            indices: [dim] or [B, dim] uint8 tensor
            scale: [] or [B] float tensor

        Returns:
            z_hat: [dim] or [B, dim] float tensor (reconstructed)
        """
        single = indices.dim() == 1
        if single:
            indices = indices.unsqueeze(0)
            scale = scale.unsqueeze(0)

        device = indices.device

        # Look up centroids
        centroids = self.centroids.to(device)
        z_scaled = centroids[indices.long()]  # [B, dim]

        # Undo coordinate scaling
        coord_scale = math.sqrt(self.dim)
        z_rot = z_scaled / coord_scale

        # Rotate back
        rot_t = self.rotation_t.to(device)
        z_norm = z_rot @ rot_t.T  # [B, dim]

        # Rescale
        if scale.dim() == 1:
            scale = scale.unsqueeze(-1)
        z_hat = z_norm * scale

        if single:
            return z_hat.squeeze(0)
        return z_hat

    def compressed_bytes_per_frame(self):
        """How many bytes per frame for this configuration."""
        # n_bits per dimension + 4 bytes for scale (float32)
        total_bits = self.dim * self.n_bits
        return math.ceil(total_bits / 8) + 4

    def bitrate_kbps(self, fps=50):
        """Bitrate in kbps at given frame rate."""
        return self.compressed_bytes_per_frame() * 8 * fps / 1000


def compute_snr(original, reconstructed, floor=0.001):
    """Signal-to-noise ratio in dB."""
    sp = (original ** 2).mean()
    if sp < floor:
        return None
    np_ = ((original - reconstructed) ** 2).mean()
    if np_ < 1e-12:
        return float('inf')
    return 10 * math.log10(max(sp.item(), 1e-10) / np_.item())


def test_turboquant():
    """Test TurboQuant on our audio latent vectors."""
    import sys
    sys.path.insert(0, '.')
    from train_audio_sppf import SPPFAudioAutoencoder
    from infer_audio import load_audio, save_wav
    from pathlib import Path

    print("=== TurboQuant for SPPF Audio ===\n")

    # Load model
    model = SPPFAudioAutoencoder(128, use_fsq=False)
    ckpt = torch.load('outputs_audio_128/resume.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('ema_shadow', ckpt.get('model_state_dict')))
    model.eval()

    # Load audio
    audio_path = Path(r'C:\Users\gcox0\OneDrive\Recordings\Greg Cox.m4a')
    waveform, sr = load_audio(audio_path)
    if sr != 16000:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform / (waveform.abs().max() + 1e-5)
    waveform = waveform[:, :30 * 16000]  # 30 seconds
    n = waveform.shape[1] // 320
    waveform = waveform[:, :n * 320]
    chunks = waveform.view(n, 1, 320)

    # Encode audio to latent vectors
    with torch.no_grad():
        all_z = torch.cat([model.encoder(chunks[i:i+64]) for i in range(0, n, 64)])

    print(f"Audio: {n} frames ({n*0.02:.1f}s)")
    print(f"Latent: {all_z.shape} (mean={all_z.mean():.3f}, std={all_z.std():.3f})")

    # Save original for reference
    save_wav(waveform, Path('infer_output/GregCox_tq_original.wav'), 16000)

    # Test raw (no compression)
    with torch.no_grad():
        raw_recon = torch.cat([model.decoder(all_z[i:i+64]) for i in range(0, n, 64)])
    raw_wav = raw_recon.squeeze(1).reshape(1, -1)
    raw_snr = compute_snr(waveform, raw_wav)
    save_wav(raw_wav, Path('infer_output/GregCox_tq_raw.wav'), 16000)

    # GRQ 8-bit baseline
    def grq(z, bits):
        s = z.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        zn = z / s
        lv = 2 ** bits
        q = (zn * (lv // 2 - 1)).round().clamp(-(lv // 2), lv // 2 - 1)
        return q / (lv // 2 - 1) * s

    z_grq8 = grq(all_z, 8)
    with torch.no_grad():
        grq8_recon = torch.cat([model.decoder(z_grq8[i:i+64]) for i in range(0, n, 64)])
    grq8_wav = grq8_recon.squeeze(1).reshape(1, -1)
    grq8_snr = compute_snr(waveform, grq8_wav)
    save_wav(grq8_wav, Path('infer_output/GregCox_tq_grq8.wav'), 16000)

    # Test TurboQuant at different bit widths
    print(f"\n{'='*65}")
    print(f"  Compression Comparison")
    print(f"{'='*65}")
    print(f"  {'Method':<25} {'SNR':>8} {'Bytes/f':>8} {'kbps':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Raw (no compress)':<25} {raw_snr:>7.1f}  {128*4:>7}  {128*4*50*8/1000:>7.1f}")
    print(f"  {'GRQ 8-bit':<25} {grq8_snr:>7.1f}  {128:>7}  {128*50*8/1000:>7.1f}")

    for n_bits in [2, 3, 4, 6, 8]:
        tq = TurboQuant(dim=128, n_bits=n_bits, seed=42)

        # Encode all frames
        indices, scales = tq.encode(all_z)

        # Decode all frames
        z_tq = tq.decode(indices, scales)

        # Reconstruct audio
        with torch.no_grad():
            tq_recon = torch.cat([model.decoder(z_tq[i:i+64]) for i in range(0, n, 64)])
        tq_wav = tq_recon.squeeze(1).reshape(1, -1)
        tq_snr = compute_snr(waveform, tq_wav)

        bpf = tq.compressed_bytes_per_frame()
        kbps = tq.bitrate_kbps()

        save_wav(tq_wav, Path(f'infer_output/GregCox_tq_{n_bits}bit.wav'), 16000)
        print(f"  {'TurboQuant ' + str(n_bits) + '-bit':<25} {tq_snr:>7.1f}  {bpf:>7}  {kbps:>7.1f}")

    print(f"  {'-'*55}")
    print(f"  {'Opus reference':<25} {'~22':>8}  {'~80':>7}  {'32.0':>8}")
    print(f"  {'Lyra reference':<25} {'~18':>8}  {'5':>7}  {'3.2':>8}")
    print(f"{'='*65}")


if __name__ == "__main__":
    test_turboquant()
