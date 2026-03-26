"""
train_learned_quant.py -- Train a neural compressor for 128-dim z-vectors.
Runs on frozen encoder/decoder. Quick experiment (~5 min on GPU).

Usage:
  python train_learned_quant.py --checkpoint outputs_audio_128/resume.pt --data_dir data
"""
import argparse, math, os, json, subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_audio_sppf import SPPFAudioAutoencoder, LibriSpeechChunks


class LearnedQuantizer(nn.Module):
    """Neural compressor: 128 floats -> N bytes -> 128 floats.
    Trained end-to-end through frozen audio decoder."""

    def __init__(self, z_dim=128, bottleneck=32, n_levels=256):
        super().__init__()
        self.bottleneck = bottleneck
        self.n_levels = n_levels
        self.enc = nn.Sequential(
            nn.Linear(z_dim, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, bottleneck), nn.Tanh(),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, z_dim),
        )

    def forward(self, z):
        compressed = self.enc(z)
        half = self.n_levels / 2
        quantized = torch.round(compressed * half) / half
        quantized_ste = compressed + (quantized - compressed).detach()
        return self.dec(quantized_ste), compressed


def compute_snr(orig, recon, floor=0.005):
    sp = (orig ** 2).mean()
    if sp < floor:
        return None
    np_ = ((orig - recon) ** 2).mean()
    if np_ < 1e-12:
        return float('inf')
    return 10 * math.log10(max(sp.item(), 1e-10) / np_.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--output_dir", default="outputs_lq")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load frozen model
    model = SPPFAudioAutoencoder(args.latent_dim, use_fsq=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("ema_shadow", ckpt.get("model_state_dict")))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Model loaded: {args.latent_dim}-dim")

    # Encode training data
    print("Loading LibriSpeech train-clean-100...")
    dataset = LibriSpeechChunks(root=args.data_dir, url="train-clean-100", download=True, crops_per_utterance=5)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    print("Loading LibriSpeech dev-clean...")
    val_dataset = LibriSpeechChunks(root=args.data_dir, url="dev-clean", download=True, crops_per_utterance=5)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # Pre-encode validation data
    print("Encoding validation data...")
    val_z = []
    val_chunks = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 20:
                break
            batch = batch.to(device)
            val_z.append(model.encoder(batch))
            val_chunks.append(batch)
    val_z = torch.cat(val_z)
    val_chunks = torch.cat(val_chunks)
    print(f"Val data: {len(val_z)} frames")

    results = {}

    for bottleneck in [16, 32, 48, 64, 96]:
        bytes_per_frame = bottleneck
        kbps = bytes_per_frame * 50 * 8 / 1000

        lq = LearnedQuantizer(z_dim=args.latent_dim, bottleneck=bottleneck).to(device)
        opt = torch.optim.Adam(lq.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

        print(f"\n{'='*60}")
        print(f"Training LearnedQ bottleneck={bottleneck} ({bytes_per_frame} bytes, {kbps:.1f} kbps)")
        print(f"{'='*60}")

        step = 0
        data_iter = iter(loader)
        while step < args.steps:
            try:
                chunks = next(data_iter).to(device)
            except StopIteration:
                data_iter = iter(loader)
                chunks = next(data_iter).to(device)

            with torch.no_grad():
                z = model.encoder(chunks)

            z_recon, _ = lq(z)

            z_loss = F.mse_loss(z_recon, z)

            # Audio-domain loss through frozen decoder
            with torch.no_grad():
                audio_orig = model.decoder(z)
            audio_recon = model.decoder(z_recon)
            audio_loss = F.mse_loss(audio_recon, audio_orig)

            loss = z_loss + 10.0 * audio_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

            if step % 500 == 0:
                print(f"  step {step}/{args.steps}: z_loss={z_loss.item():.6f} audio_loss={audio_loss.item():.6f}")
            step += 1

        # Evaluate
        lq.eval()
        snr_sum = 0
        snr_count = 0
        with torch.no_grad():
            z_recon, _ = lq(val_z)
            for i in range(0, len(val_z), 64):
                orig_audio = model.decoder(val_z[i:i+64])
                recon_audio = model.decoder(z_recon[i:i+64])
                for j in range(orig_audio.shape[0]):
                    s = compute_snr(orig_audio[j], recon_audio[j])
                    if s is not None:
                        snr_sum += s
                        snr_count += 1

        avg_snr = snr_sum / max(snr_count, 1)
        print(f"  -> SNR={avg_snr:.1f} dB | {bytes_per_frame} bytes/frame | {kbps:.1f} kbps")

        results[bottleneck] = {"snr": avg_snr, "bytes": bytes_per_frame, "kbps": kbps}

        # Save quantizer
        torch.save(lq.state_dict(), os.path.join(args.output_dir, f"lq_{bottleneck}.pt"))

    # Save results
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Raw (no compress): ~25.8 dB | 512 bytes | 204.8 kbps")
    print(f"  GRQ 8-bit:         25.8 dB | 128 bytes | 51.2 kbps")
    for b, r in sorted(results.items()):
        print(f"  LearnedQ b={b}:     {r['snr']:.1f} dB | {r['bytes']} bytes | {r['kbps']:.1f} kbps")
    print(f"  Opus:              ~22 dB  | ~80 bytes | 32 kbps")
    print(f"  Lyra:              ~18 dB  | 5 bytes   | 3.2 kbps")

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Upload if on RunPod
    gh_repo = os.environ.get("GH_REPO", "")
    release_tag = os.environ.get("RELEASE_TAG", "")
    if gh_repo and release_tag:
        for f in Path(args.output_dir).glob("*.pt"):
            subprocess.run(["gh", "release", "upload", release_tag, str(f), "--repo", gh_repo, "--clobber"], capture_output=True)
        subprocess.run(["gh", "release", "upload", release_tag, os.path.join(args.output_dir, "results.json"), "--repo", gh_repo, "--clobber"], capture_output=True)


if __name__ == "__main__":
    main()
