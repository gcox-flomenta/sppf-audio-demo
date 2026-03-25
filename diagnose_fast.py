"""
Fast diagnostic tests — CPU-friendly, small batches, fewer steps.
"""
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"C:\code\sppf_audio_demo")
from train_audio_sppf import (
    SPPFAudioAutoencoder, Encoder, Decoder, FSQQuantizer,
    MultiResolutionSTFTLoss, MelSpectrogramLoss, compute_snr
)

torch.manual_seed(42)
device = torch.device("cpu")  # Force CPU for speed on this machine
print(f"Device: {device}")
B = 8  # small batch

def make_audio(batch_size=8):
    t = torch.linspace(0, 0.02, 320).unsqueeze(0).expand(batch_size, -1)
    freqs = torch.randint(100, 4000, (batch_size, 1)).float()
    audio = torch.sin(2 * math.pi * freqs * t)
    audio += 0.5 * torch.sin(2 * math.pi * 2 * freqs * t)
    audio += 0.1 * torch.randn_like(audio)
    audio = audio / (audio.abs().max(dim=1, keepdim=True)[0] + 1e-5)
    return audio.unsqueeze(1)  # [B, 1, 320]

audio = make_audio(B)

# ============================================================
print("\n" + "=" * 70)
print("TEST 1: FSQ vs No-FSQ forward pass (untrained)")
print("=" * 70)
model = SPPFAudioAutoencoder(512)
with torch.no_grad():
    z = model.encoder(audio)
    z_q, z_cont = model.quantizer(z)
    recon_fsq = model.decoder(z_q)
    recon_no = model.decoder(z)
    mse_fsq = F.mse_loss(recon_fsq, audio).item()
    mse_no = F.mse_loss(recon_no, audio).item()
print(f"  MSE with FSQ:    {mse_fsq:.6f}")
print(f"  MSE without FSQ: {mse_no:.6f}")
print(f"  Ratio: {mse_fsq/mse_no:.2f}x")

# ============================================================
print("\n" + "=" * 70)
print("TEST 2: FSQ information destruction")
print("=" * 70)
q = model.quantizer
with torch.no_grad():
    z_enc = model.encoder(audio)
    z_q_enc, _ = q(z_enc)
    mse_fsq_enc = F.mse_loss(z_q_enc, z_enc).item()
    print(f"  Encoder output -> FSQ -> MSE: {mse_fsq_enc:.6f}")
    print(f"  z_enc: mean={z_enc.mean():.4f}, std={z_enc.std():.4f}, min={z_enc.min():.4f}, max={z_enc.max():.4f}")
    print(f"  z_q:   mean={z_q_enc.mean():.4f}, std={z_q_enc.std():.4f}, min={z_q_enc.min():.4f}, max={z_q_enc.max():.4f}")

    z_low = q.project_down(z_enc)
    z_bounded = q._bound(z_low)
    z_hat = torch.round(z_bounded)
    print(f"\n  FSQ internals:")
    print(f"    project_down: mean={z_low.mean():.4f}, std={z_low.std():.4f}, range=[{z_low.min():.4f}, {z_low.max():.4f}]")
    print(f"    after bound:  mean={z_bounded.mean():.4f}, std={z_bounded.std():.4f}, range=[{z_bounded.min():.2f}, {z_bounded.max():.2f}]")
    print(f"    after round:  unique levels = {sorted(set(z_hat.flatten().numpy().tolist()))}")
    print(f"    rounding err: {(z_bounded - z_hat).abs().mean():.4f}")

    # The REAL question: what fraction of 512 dims can be reconstructed from 32?
    # project_up reconstructs 512 from 32 — this is a 16x compression of the latent!
    print(f"\n  CRITICAL: Latent bottleneck = 512 -> 32 -> 512")
    print(f"    That's 16x compression of the latent space!")
    print(f"    Even without rounding, project_down destroys rank:")
    z_reconstruct_no_round = q.project_up(q.project_down(z_enc))
    mse_no_round = F.mse_loss(z_reconstruct_no_round, z_enc).item()
    print(f"    MSE without rounding (just project_down -> project_up): {mse_no_round:.6f}")
    print(f"    MSE WITH rounding: {mse_fsq_enc:.6f}")
    print(f"    -> Rounding adds {((mse_fsq_enc/mse_no_round - 1)*100):.0f}% more error on top of projection loss")

# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Encoder+Decoder overfit (NO FSQ) — 100 steps")
print("=" * 70)
model2 = SPPFAudioAutoencoder(512)
opt2 = torch.optim.Adam(list(model2.encoder.parameters()) + list(model2.decoder.parameters()), lr=1e-3)
for step in range(100):
    z = model2.encoder(audio)
    recon = model2.decoder(z)
    loss = F.mse_loss(recon, audio)
    opt2.zero_grad()
    loss.backward()
    opt2.step()
    if (step+1) % 25 == 0:
        with torch.no_grad():
            snr = compute_snr(audio, recon)
            print(f"  Step {step+1}: MSE={loss.item():.6f}, SNR={snr:.1f} dB" if snr else f"  Step {step+1}: MSE={loss.item():.6f}")

# ============================================================
print("\n" + "=" * 70)
print("TEST 4: Full model WITH FSQ overfit — 200 steps")
print("=" * 70)
model3 = SPPFAudioAutoencoder(512)
opt3 = torch.optim.Adam(model3.parameters(), lr=1e-3)
for step in range(200):
    recon, zc, zq = model3(audio)
    loss = F.mse_loss(recon, audio)
    opt3.zero_grad()
    loss.backward()
    opt3.step()
    if (step+1) % 50 == 0:
        with torch.no_grad():
            snr = compute_snr(audio, recon)
            print(f"  Step {step+1}: MSE={loss.item():.6f}, SNR={snr:.1f} dB" if snr else f"  Step {step+1}: MSE={loss.item():.6f}")

# ============================================================
print("\n" + "=" * 70)
print("TEST 5: Gradient norms — FSQ vs no-FSQ")
print("=" * 70)
model4 = SPPFAudioAutoencoder(512)

# With FSQ
recon4, zc4, zq4 = model4(audio)
loss4 = F.mse_loss(recon4, audio)
loss4.backward()
gn_enc_fsq = sum(p.grad.norm().item()**2 for p in model4.encoder.parameters() if p.grad is not None)**0.5
gn_dec_fsq = sum(p.grad.norm().item()**2 for p in model4.decoder.parameters() if p.grad is not None)**0.5
gn_fsq = sum(p.grad.norm().item()**2 for p in model4.quantizer.parameters() if p.grad is not None)**0.5
model4.zero_grad()

# Without FSQ
z4 = model4.encoder(audio)
recon4b = model4.decoder(z4)
loss4b = F.mse_loss(recon4b, audio)
loss4b.backward()
gn_enc_no = sum(p.grad.norm().item()**2 for p in model4.encoder.parameters() if p.grad is not None)**0.5
gn_dec_no = sum(p.grad.norm().item()**2 for p in model4.decoder.parameters() if p.grad is not None)**0.5

print(f"  WITH FSQ:    enc_grad={gn_enc_fsq:.6f}, dec_grad={gn_dec_fsq:.6f}, fsq_grad={gn_fsq:.6f}")
print(f"  WITHOUT FSQ: enc_grad={gn_enc_no:.6f}, dec_grad={gn_dec_no:.6f}")
print(f"  Encoder grad ratio: {gn_enc_fsq/(gn_enc_no+1e-10):.4f}")

# ============================================================
print("\n" + "=" * 70)
print("TEST 6: Loss component magnitudes")
print("=" * 70)
model5 = SPPFAudioAutoencoder(512)
stft_fn = MultiResolutionSTFTLoss(fft_sizes=(64, 128, 256))
mel_fn = MelSpectrogramLoss(sr=16000, n_fft=256, hop_length=64, n_mels=64)

with torch.no_grad():
    recon5, zc5, zq5 = model5(audio)
    mse = F.mse_loss(recon5, audio)
    stft = stft_fn(recon5, audio)
    mel = mel_fn(recon5, audio)
    quant = F.mse_loss(zc5, zq5.detach())
    total = 10*mse + stft + mel + 0.1*quant
print(f"  10*MSE:  {10*mse.item():.4f}  ({10*mse.item()/total.item()*100:.1f}%)")
print(f"  STFT:    {stft.item():.4f}  ({stft.item()/total.item()*100:.1f}%)")
print(f"  Mel:     {mel.item():.4f}  ({mel.item()/total.item()*100:.1f}%)")
print(f"  0.1*Qnt: {0.1*quant.item():.4f}  ({0.1*quant.item()/total.item()*100:.1f}%)")
print(f"  Total:   {total.item():.4f}")

# ============================================================
print("\n" + "=" * 70)
print("TEST 7: Decoder clamp analysis")
print("=" * 70)
with torch.no_grad():
    model6 = SPPFAudioAutoencoder(512)
    z6 = model6.encoder(audio)
    x6 = z6.unsqueeze(-1)
    x6 = model6.decoder.block1(x6)
    x6 = model6.decoder.block2(x6)
    x6 = model6.decoder.block3(x6)
    x6 = model6.decoder.block4(x6)
    pre_clamp = model6.decoder.final(x6)
    pct = (pre_clamp.abs() > 1.0).float().mean().item() * 100
    print(f"  Pre-clamp range: [{pre_clamp.min():.4f}, {pre_clamp.max():.4f}]")
    print(f"  % clamped: {pct:.1f}%")

# ============================================================
print("\n" + "=" * 70)
print("TEST 8: CRITICAL — Generalization test (20 batches x 3 epochs)")
print("=" * 70)

print("\n  A) WITHOUT FSQ:")
model7a = SPPFAudioAutoencoder(512)
opt7a = torch.optim.Adam(list(model7a.encoder.parameters()) + list(model7a.decoder.parameters()), lr=1e-3)
for epoch in range(3):
    eloss = 0
    for b in range(20):
        batch = make_audio(B)
        z = model7a.encoder(batch)
        recon = model7a.decoder(z)
        loss = F.mse_loss(recon, batch)
        opt7a.zero_grad(); loss.backward(); opt7a.step()
        eloss += loss.item()
    with torch.no_grad():
        tb = make_audio(B)
        tr = model7a.decoder(model7a.encoder(tb))
        tm = F.mse_loss(tr, tb).item()
        ts = compute_snr(tb, tr)
        print(f"    Epoch {epoch+1}: train_mse={eloss/20:.6f}, test_mse={tm:.6f}, test_SNR={ts:.1f} dB" if ts else f"    Epoch {epoch+1}: train_mse={eloss/20:.6f}, test_mse={tm:.6f}")

print("\n  B) WITH FSQ:")
model7b = SPPFAudioAutoencoder(512)
opt7b = torch.optim.Adam(model7b.parameters(), lr=1e-3)
for epoch in range(3):
    eloss = 0
    for b in range(20):
        batch = make_audio(B)
        recon, zc, zq = model7b(batch)
        loss = F.mse_loss(recon, batch)
        opt7b.zero_grad(); loss.backward(); opt7b.step()
        eloss += loss.item()
    with torch.no_grad():
        tb = make_audio(B)
        tr, _, _ = model7b(tb)
        tm = F.mse_loss(tr, tb).item()
        ts = compute_snr(tb, tr)
        print(f"    Epoch {epoch+1}: train_mse={eloss/20:.6f}, test_mse={tm:.6f}, test_SNR={ts:.1f} dB" if ts else f"    Epoch {epoch+1}: train_mse={eloss/20:.6f}, test_mse={tm:.6f}")

# ============================================================
print("\n" + "=" * 70)
print("TEST 9: MSE vs STFT gradient conflict check")
print("=" * 70)
model8 = SPPFAudioAutoencoder(512)
stft_fn2 = MultiResolutionSTFTLoss(fft_sizes=(64, 128, 256))
a9 = make_audio(B)

recon8, _, _ = model8(a9)
mse8 = F.mse_loss(recon8, a9)
model8.zero_grad()
mse8.backward(retain_graph=True)
grad_mse = {}
for n, p in model8.named_parameters():
    if p.grad is not None:
        grad_mse[n] = p.grad.clone()

model8.zero_grad()
stft8 = stft_fn2(recon8, a9)
stft8.backward()
cos_sims = []
for n, p in model8.named_parameters():
    if p.grad is not None and n in grad_mse:
        g1 = grad_mse[n].flatten()
        g2 = p.grad.flatten()
        if g1.norm() > 1e-10 and g2.norm() > 1e-10:
            cos = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
            cos_sims.append((n, cos))

cos_sims.sort(key=lambda x: x[1])
print("  Most conflicting:")
for n, c in cos_sims[:5]:
    print(f"    {n}: {c:.4f} {'CONFLICT' if c < 0 else 'ok'}")
avg_cos = sum(c for _, c in cos_sims) / len(cos_sims) if cos_sims else 0
print(f"  Avg cosine sim: {avg_cos:.4f}")

# ============================================================
print("\n" + "=" * 70)
print("TEST 10: What happens with MORE FSQ dims (64 instead of 32)?")
print("=" * 70)

class SPPFAudioAutoencoder64(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(512)
        self.quantizer = FSQQuantizer(latent_dim=512, levels=[5]*64)  # 64 dims instead of 32
        self.decoder = Decoder(512)
    def forward(self, x):
        z = self.encoder(x)
        z_q, z_cont = self.quantizer(z)
        recon = self.decoder(z_q)
        return recon, z_cont, z_q

model9 = SPPFAudioAutoencoder64()
opt9 = torch.optim.Adam(model9.parameters(), lr=1e-3)
for step in range(200):
    recon9, zc9, zq9 = model9(audio)
    loss = F.mse_loss(recon9, audio)
    opt9.zero_grad(); loss.backward(); opt9.step()
    if (step+1) % 50 == 0:
        with torch.no_grad():
            snr = compute_snr(audio, recon9)
            print(f"  Step {step+1}: MSE={loss.item():.6f}, SNR={snr:.1f} dB" if snr else f"  Step {step+1}: MSE={loss.item():.6f}")

# Compare FSQ reconstruction with 64 dims
with torch.no_grad():
    z9 = model9.encoder(audio)
    z_q9, _ = model9.quantizer(z9)
    mse_64 = F.mse_loss(z_q9, z9).item()

    z_no_round9 = model9.quantizer.project_up(model9.quantizer.project_down(z9))
    mse_64_no_round = F.mse_loss(z_no_round9, z9).item()

print(f"\n  FSQ 64 dims: projection MSE={mse_64_no_round:.6f}, with rounding MSE={mse_64:.6f}")
print(f"  FSQ 32 dims: projection MSE={mse_no_round:.6f} (from Test 2)")
print(f"  -> 64 dims reduces projection error by {(1-mse_64_no_round/mse_no_round)*100:.0f}%")


# ============================================================
print("\n" + "=" * 70)
print("TEST 11: Is the 320:1 temporal compression the issue?")
print("    Test: larger input (640 samples, stride=160) vs 320")
print("=" * 70)

# Can't easily change architecture, but we can test: what if we encode TWO
# adjacent 320-sample frames and concatenate their latents?
# This gives 1024-dim latent for 640 samples = 640:2 = 320:1 same ratio
# But the decoder sees a richer latent...
# Actually let's just test: what MSE does the encoder achieve on single pass?
with torch.no_grad():
    model_t = SPPFAudioAutoencoder(512)
    a_t = make_audio(B)
    z_t = model_t.encoder(a_t)
    print(f"  Encoder output (z) shape: {z_t.shape}")
    print(f"  z stats: mean={z_t.mean():.4f}, std={z_t.std():.4f}")
    print(f"  320 samples compressed to {z_t.shape[1]} floats = {320/z_t.shape[1]:.1f}:1 ratio")
    print(f"  With FSQ 32 dims = 320 samples to 32 quantized values = 10:1 samples-per-dim")


# ============================================================
print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)
