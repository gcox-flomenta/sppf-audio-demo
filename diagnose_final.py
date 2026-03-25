"""
Final targeted diagnostic: the FSQ project_up is AMPLIFYING latents by 13x.
z_enc has std=0.075, but z_q has std=0.99. The decoder sees wildly different
magnitudes than what the encoder produces.

Also: MSE and STFT have avg cosine sim = -0.175 (they FIGHT each other).
"""
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"C:\code\sppf_audio_demo")
from train_audio_sppf import (
    SPPFAudioAutoencoder, Encoder, Decoder, FSQQuantizer, compute_snr
)

torch.manual_seed(42)
B = 8

def make_audio(batch_size=8):
    t = torch.linspace(0, 0.02, 320).unsqueeze(0).expand(batch_size, -1)
    freqs = torch.randint(100, 4000, (batch_size, 1)).float()
    audio = torch.sin(2 * math.pi * freqs * t)
    audio += 0.5 * torch.sin(2 * math.pi * 2 * freqs * t)
    audio += 0.1 * torch.randn_like(audio)
    audio = audio / (audio.abs().max(dim=1, keepdim=True)[0] + 1e-5)
    return audio.unsqueeze(1)

audio = make_audio(B)

# ============================================================
print("=" * 70)
print("TARGETED TEST A: FSQ scale mismatch")
print("=" * 70)

model = SPPFAudioAutoencoder(512)
with torch.no_grad():
    z = model.encoder(audio)
    z_q, _ = model.quantizer(z)
    print(f"Encoder output z: std={z.std():.4f}, mean={z.mean():.4f}")
    print(f"FSQ output z_q:   std={z_q.std():.4f}, mean={z_q.mean():.4f}")
    print(f"Amplification factor: {z_q.std() / z.std():.1f}x")
    print(f"\nThe decoder was designed to receive encoder outputs (std~0.075)")
    print(f"But FSQ project_up sends it values with std~1.0")
    print(f"This is a SCALE MISMATCH of ~13x!")

# ============================================================
print("\n" + "=" * 70)
print("TARGETED TEST B: What if we fix the scale mismatch?")
print("  Scale z_q to match z's statistics before feeding to decoder")
print("=" * 70)

class FixedScaleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(512)
        self.quantizer = FSQQuantizer(latent_dim=512)
        self.decoder = Decoder(512)

    def forward(self, x):
        z = self.encoder(x)
        z_q, z_cont = self.quantizer(z)
        # Scale z_q to match z's statistics
        z_q_scaled = z_q * (z.std() / (z_q.std() + 1e-6))
        recon = self.decoder(z_q_scaled)
        return recon, z_cont, z_q

model_fix = FixedScaleModel()
opt_fix = torch.optim.Adam(model_fix.parameters(), lr=1e-3)
for step in range(200):
    recon, zc, zq = model_fix(audio)
    loss = F.mse_loss(recon, audio)
    opt_fix.zero_grad(); loss.backward(); opt_fix.step()
    if (step+1) % 50 == 0:
        with torch.no_grad():
            snr = compute_snr(audio, recon)
            z = model_fix.encoder(audio)
            zq_out, _ = model_fix.quantizer(z)
            print(f"  Step {step+1}: MSE={loss.item():.6f}, SNR={snr:.1f} dB, z_std={z.std():.4f}, zq_std={zq_out.std():.4f}")

# ============================================================
print("\n" + "=" * 70)
print("TARGETED TEST C: Train WITHOUT STFT loss (only MSE)")
print("  Since MSE and STFT fight each other (avg cos sim = -0.175)")
print("=" * 70)

model_mse = SPPFAudioAutoencoder(512)
opt_mse = torch.optim.Adam(model_mse.parameters(), lr=1e-3)
for step in range(200):
    recon, zc, zq = model_mse(audio)
    loss = F.mse_loss(recon, audio)  # MSE ONLY
    opt_mse.zero_grad(); loss.backward(); opt_mse.step()
    if (step+1) % 50 == 0:
        with torch.no_grad():
            snr = compute_snr(audio, recon)
            print(f"  Step {step+1}: MSE={loss.item():.6f}, SNR={snr:.1f} dB")

print("\n  Compare with Test 4 (full model, MSE only): stuck at 0.7 dB")
print("  If this is ALSO stuck, the problem isn't STFT conflict alone")

# ============================================================
print("\n" + "=" * 70)
print("TARGETED TEST D: Generalization WITHOUT FSQ (more epochs)")
print("=" * 70)

model_gen = SPPFAudioAutoencoder(512)
opt_gen = torch.optim.Adam(
    list(model_gen.encoder.parameters()) + list(model_gen.decoder.parameters()),
    lr=3e-4
)

for epoch in range(10):
    eloss = 0
    for b in range(50):
        batch = make_audio(B)
        z = model_gen.encoder(batch)
        recon = model_gen.decoder(z)
        loss = F.mse_loss(recon, batch)
        opt_gen.zero_grad(); loss.backward(); opt_gen.step()
        eloss += loss.item()
    with torch.no_grad():
        tb = make_audio(32)
        tr = model_gen.decoder(model_gen.encoder(tb))
        tm = F.mse_loss(tr, tb).item()
        ts = compute_snr(tb, tr)
        print(f"  Epoch {epoch+1}: train_mse={eloss/50:.6f}, test_mse={tm:.6f}, test_SNR={ts:.1f} dB")

# ============================================================
print("\n" + "=" * 70)
print("TARGETED TEST E: Check project_up weight norms")
print("=" * 70)

q = model.quantizer
print(f"  project_down weight: shape={q.project_down.weight.shape}, norm={q.project_down.weight.norm():.4f}")
print(f"  project_up weight:   shape={q.project_up.weight.shape}, norm={q.project_up.weight.norm():.4f}")
print(f"  project_down weight std: {q.project_down.weight.std():.4f}")
print(f"  project_up weight std:   {q.project_up.weight.std():.4f}")
print(f"\n  project_down was initialized with std=2.0 (intentionally large)")
print(f"  project_up uses default init (std~{1/math.sqrt(32):.4f} for fan_in=32)")
print(f"  But iFSQ maps to [-2, 2], so project_up(round(bound(x))) has")
print(f"  inputs in [-2, 2] with 32 dims -> output magnitude = ~32 * std * 2")

# What's the expected output magnitude?
import numpy as np
# project_up has shape [512, 32], default init ~ N(0, 1/sqrt(32))
# Input is 32 values each in {-2, -1, 0, 1, 2}
# Output = W @ x, each output dim = sum of 32 terms, each ~ w_i * x_i
# Var(output) = 32 * Var(w) * Var(x) = 32 * (1/32) * Var(x)
# If x uniform in {-2,-1,0,1,2}: Var(x) = E[x^2] = (4+1+0+1+4)/5 = 2.0
# So Var(output) = 32 * (1/32) * 2.0 = 2.0, std(output) = 1.41
print(f"\n  Expected z_q std (theoretical): ~1.41")
print(f"  Actual z_q std: {model.quantizer.project_up(torch.tensor([[0.0]*32])).std():.4f} (for zero input)")
# For realistic input:
test_input = torch.randint(-2, 3, (100, 32)).float()
test_output = q.project_up(test_input)
print(f"  Actual z_q std (random quantized input): {test_output.std():.4f}")
print(f"  Encoder output std: ~0.075")
print(f"  -> FSQ output is {test_output.std()/0.075:.0f}x larger than encoder output!")

print("\n" + "=" * 70)
print("ROOT CAUSE ANALYSIS")
print("=" * 70)
print("""
THREE ROOT CAUSES FOUND:

1. FSQ SCALE MISMATCH (CRITICAL):
   - Encoder outputs z with std ~0.075
   - FSQ project_up outputs z_q with std ~1.0
   - Decoder receives 13x larger inputs than the encoder would produce
   - During single-batch overfitting, the model CAN adapt (learns to
     scale down in decoder), but on diverse data this compensation fails
   - FIX: Add LayerNorm before FSQ, or normalize z_q to match z stats,
     or init project_up to preserve scale

2. MSE vs STFT GRADIENT CONFLICT (SIGNIFICANT):
   - Average cosine similarity = -0.175 (gradients point in OPPOSITE directions!)
   - decoder.final has cos_sim = -1.0 (perfectly opposed!)
   - The two losses literally fight, causing oscillation
   - FIX: Use only MSE for the first N epochs, then add STFT gradually.
     Or reduce STFT weight. Or use a combined spectral loss that agrees
     with MSE direction (like multi-resolution STFT with spectral
     convergence only, no log magnitude).

3. FSQ BOTTLENECK IS MODERATE (not the main issue):
   - 512->32->512 projection alone has MSE ~3.7 (without rounding)
   - But rounding REDUCES this to 0.98 (because project_up was randomly
     init'd and rounding happens to project to a more "normal" distribution)
   - The real issue isn't information destruction per se, but the SCALE
     of the reconstructed values

PREDICTED FIX ORDER OF IMPACT:
  1. Fix scale mismatch -> expect 10-15 dB improvement
  2. Fix loss conflict -> expect 3-5 dB improvement
  3. Increase FSQ dims (32->64) -> expect 1-2 dB improvement
""")
