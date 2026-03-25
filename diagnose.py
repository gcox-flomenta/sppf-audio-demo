"""
Diagnostic tests to find the root cause of -1 dB SNR plateau.
Tests FSQ bottleneck, encoder-decoder capacity, gradients, loss weights, and data distribution.
"""
import sys
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import everything from the training script
sys.path.insert(0, r"C:\code\sppf_audio_demo")
from train_audio_sppf import (
    SPPFAudioAutoencoder, Encoder, Decoder, FSQQuantizer,
    MultiResolutionSTFTLoss, MelSpectrogramLoss, compute_snr
)

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("=" * 70)

# Generate realistic audio-like data (sine waves + noise, normalized to [-1,1])
def make_fake_audio(batch_size=64):
    """Simulate speech-like audio: mix of frequencies + noise"""
    t = torch.linspace(0, 0.02, 320).unsqueeze(0).expand(batch_size, -1)
    freqs = torch.randint(100, 4000, (batch_size, 1)).float()
    audio = torch.sin(2 * math.pi * freqs * t)
    # Add harmonics
    audio += 0.5 * torch.sin(2 * math.pi * 2 * freqs * t)
    audio += 0.3 * torch.sin(2 * math.pi * 3 * freqs * t)
    # Add noise
    audio += 0.1 * torch.randn_like(audio)
    # Normalize per sample
    audio = audio / (audio.abs().max(dim=1, keepdim=True)[0] + 1e-5)
    return audio.unsqueeze(1).to(device)  # [B, 1, 320]


# ============================================================
# TEST 1: FSQ vs No-FSQ forward pass comparison
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: FSQ vs No-FSQ forward pass (untrained model)")
print("=" * 70)

model_full = SPPFAudioAutoencoder(512).to(device)
audio = make_fake_audio(64)

with torch.no_grad():
    # With FSQ
    z = model_full.encoder(audio)
    z_q, z_cont = model_full.quantizer(z)
    recon_fsq = model_full.decoder(z_q)
    mse_fsq = F.mse_loss(recon_fsq, audio).item()

    # Without FSQ (bypass quantizer)
    recon_no_fsq = model_full.decoder(z)
    mse_no_fsq = F.mse_loss(recon_no_fsq, audio).item()

print(f"  MSE with FSQ:    {mse_fsq:.6f}")
print(f"  MSE without FSQ: {mse_no_fsq:.6f}")
print(f"  Ratio (fsq/no_fsq): {mse_fsq/mse_no_fsq:.2f}x")
print(f"  -> {'FSQ is a major bottleneck' if mse_fsq/mse_no_fsq > 2 else 'FSQ is NOT the main bottleneck (similar MSE)'}")


# ============================================================
# TEST 2: FSQ reconstruction quality directly
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: FSQ information destruction (512-dim -> 32-dim -> 512-dim)")
print("=" * 70)

quantizer = model_full.quantizer
with torch.no_grad():
    # Random encoder output (typical range)
    z_random = torch.randn(1000, 512).to(device)
    z_q_out, _ = quantizer(z_random)
    mse_fsq_direct = F.mse_loss(z_q_out, z_random).item()

    # Also test with actual encoder outputs
    z_enc = model_full.encoder(audio)
    z_q_enc, _ = quantizer(z_enc)
    mse_fsq_enc = F.mse_loss(z_q_enc, z_enc).item()

    # What's the variance of z_enc?
    z_enc_var = z_enc.var().item()
    z_enc_mean = z_enc.mean().item()
    z_enc_min = z_enc.min().item()
    z_enc_max = z_enc.max().item()

print(f"  FSQ reconstruction MSE (random 512-d vectors): {mse_fsq_direct:.6f}")
print(f"  FSQ reconstruction MSE (actual encoder outputs): {mse_fsq_enc:.6f}")
print(f"  Encoder output stats: mean={z_enc_mean:.4f}, var={z_enc_var:.4f}, min={z_enc_min:.4f}, max={z_enc_max:.4f}")
print(f"  Relative error: {mse_fsq_enc / (z_enc_var + 1e-8):.4f} (MSE / variance)")

# Check what FSQ actually preserves
with torch.no_grad():
    z_low = quantizer.project_down(z_enc)  # [B, 32]
    z_bounded = quantizer._bound(z_low)
    z_hat = torch.round(z_bounded)
    print(f"\n  FSQ internal stats:")
    print(f"    project_down output: mean={z_low.mean():.4f}, std={z_low.std():.4f}, min={z_low.min():.4f}, max={z_low.max():.4f}")
    print(f"    After bound (iFSQ):  mean={z_bounded.mean():.4f}, std={z_bounded.std():.4f}, min={z_bounded.min():.4f}, max={z_bounded.max():.4f}")
    print(f"    After round:         mean={z_hat.mean():.4f}, std={z_hat.std():.4f}")
    print(f"    Rounding error:      {(z_bounded - z_hat).abs().mean():.4f}")
    # Check level utilization
    unique_levels = set()
    for i in range(32):
        unique_levels.update(z_hat[:, i].unique().cpu().numpy().tolist())
    print(f"    Unique quantization levels used: {sorted(unique_levels)}")


# ============================================================
# TEST 3: Encoder-Decoder without FSQ — can it reconstruct?
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Encoder+Decoder (NO FSQ) — single batch overfit test")
print("=" * 70)

model_nofsq = SPPFAudioAutoencoder(512).to(device)
audio_batch = make_fake_audio(32)

optimizer_test = torch.optim.Adam(
    list(model_nofsq.encoder.parameters()) + list(model_nofsq.decoder.parameters()),
    lr=1e-3
)

for step in range(200):
    z = model_nofsq.encoder(audio_batch)
    recon = model_nofsq.decoder(z)  # bypass FSQ
    loss = F.mse_loss(recon, audio_batch)
    optimizer_test.zero_grad()
    loss.backward()
    optimizer_test.step()

    if (step + 1) % 50 == 0:
        with torch.no_grad():
            snr = compute_snr(audio_batch, recon)
            snr_str = f"{snr:.1f}" if snr is not None else "N/A"
        print(f"  Step {step+1}: MSE={loss.item():.6f}, SNR={snr_str} dB")

print(f"  -> Encoder+Decoder CAN overfit without FSQ: {snr_str} dB")


# ============================================================
# TEST 4: Full model WITH FSQ — single batch overfit test
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: Full model WITH FSQ — single batch overfit test")
print("=" * 70)

model_fsq = SPPFAudioAutoencoder(512).to(device)
optimizer_test2 = torch.optim.Adam(model_fsq.parameters(), lr=1e-3)

for step in range(500):
    recon, z_cont, z_q = model_fsq(audio_batch)
    loss = F.mse_loss(recon, audio_batch)
    optimizer_test2.zero_grad()
    loss.backward()
    optimizer_test2.step()

    if (step + 1) % 100 == 0:
        with torch.no_grad():
            snr = compute_snr(audio_batch, recon)
            snr_str = f"{snr:.1f}" if snr is not None else "N/A"
        print(f"  Step {step+1}: MSE={loss.item():.6f}, SNR={snr_str} dB")


# ============================================================
# TEST 5: Gradient norms — FSQ vs no-FSQ
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: Gradient norms comparison (FSQ vs no-FSQ)")
print("=" * 70)

model_a = SPPFAudioAutoencoder(512).to(device)
audio_grad = make_fake_audio(32)

# With FSQ
recon_a, z_cont_a, z_q_a = model_a(audio_grad)
loss_a = F.mse_loss(recon_a, audio_grad)
loss_a.backward()

grad_norm_enc_fsq = sum(p.grad.norm().item()**2 for p in model_a.encoder.parameters() if p.grad is not None)**0.5
grad_norm_dec_fsq = sum(p.grad.norm().item()**2 for p in model_a.decoder.parameters() if p.grad is not None)**0.5
grad_norm_fsq_mod = sum(p.grad.norm().item()**2 for p in model_a.quantizer.parameters() if p.grad is not None)**0.5

model_a.zero_grad()

# Without FSQ
z_b = model_a.encoder(audio_grad)
recon_b = model_a.decoder(z_b)
loss_b = F.mse_loss(recon_b, audio_grad)
loss_b.backward()

grad_norm_enc_no = sum(p.grad.norm().item()**2 for p in model_a.encoder.parameters() if p.grad is not None)**0.5
grad_norm_dec_no = sum(p.grad.norm().item()**2 for p in model_a.decoder.parameters() if p.grad is not None)**0.5

print(f"  WITH FSQ:    encoder_grad={grad_norm_enc_fsq:.6f}, decoder_grad={grad_norm_dec_fsq:.6f}, fsq_grad={grad_norm_fsq_mod:.6f}")
print(f"  WITHOUT FSQ: encoder_grad={grad_norm_enc_no:.6f}, decoder_grad={grad_norm_dec_no:.6f}")
print(f"  Encoder grad ratio (fsq/no_fsq): {grad_norm_enc_fsq/(grad_norm_enc_no+1e-10):.4f}")
print(f"  Decoder grad ratio (fsq/no_fsq): {grad_norm_dec_fsq/(grad_norm_dec_no+1e-10):.4f}")


# ============================================================
# TEST 6: Loss component magnitudes at initialization
# ============================================================
print("\n" + "=" * 70)
print("TEST 6: Loss component magnitudes at initialization")
print("=" * 70)

model_c = SPPFAudioAutoencoder(512).to(device)
stft_loss_fn = MultiResolutionSTFTLoss(fft_sizes=(64, 128, 256)).to(device)
mel_loss_fn = MelSpectrogramLoss(sr=16000, n_fft=256, hop_length=64, n_mels=64).to(device)

with torch.no_grad():
    recon_c, z_cont_c, z_q_c = model_c(audio_grad)
    mse = F.mse_loss(recon_c, audio_grad)
    stft = stft_loss_fn(recon_c, audio_grad)
    mel = mel_loss_fn(recon_c, audio_grad)
    quant = F.mse_loss(z_cont_c, z_q_c.detach())

    total = 10.0 * mse + stft + 1.0 * mel + 0.1 * quant

print(f"  Raw MSE:        {mse.item():.6f}  (weighted: {10*mse.item():.4f})")
print(f"  STFT loss:      {stft.item():.6f}  (weighted: {stft.item():.4f})")
print(f"  Mel loss:       {mel.item():.6f}  (weighted: {mel.item():.4f})")
print(f"  Quant loss:     {quant.item():.6f}  (weighted: {0.1*quant.item():.4f})")
print(f"  Total:          {total.item():.4f}")
print(f"\n  Fraction of total:")
print(f"    10*MSE:  {10*mse.item()/total.item()*100:.1f}%")
print(f"    STFT:    {stft.item()/total.item()*100:.1f}%")
print(f"    Mel:     {mel.item()/total.item()*100:.1f}%")
print(f"    Quant:   {0.1*quant.item()/total.item()*100:.1f}%")


# ============================================================
# TEST 7: Data distribution (using synthetic + check normalization)
# ============================================================
print("\n" + "=" * 70)
print("TEST 7: Audio data distribution analysis")
print("=" * 70)

audio_test = make_fake_audio(256)
rms = audio_test.squeeze(1).pow(2).mean(dim=1).sqrt()
print(f"  Synthetic audio stats:")
print(f"    Shape: {audio_test.shape}")
print(f"    RMS: mean={rms.mean():.4f}, min={rms.min():.4f}, max={rms.max():.4f}")
print(f"    Signal range: [{audio_test.min():.4f}, {audio_test.max():.4f}]")
print(f"    Near-zero (|x| < 0.01): {(audio_test.abs() < 0.01).float().mean()*100:.1f}%")

# Check what model outputs look like at init
with torch.no_grad():
    model_d = SPPFAudioAutoencoder(512).to(device)
    recon_d, _, _ = model_d(audio_test[:16])
    recon_rms = recon_d.squeeze(1).pow(2).mean(dim=1).sqrt()
    print(f"\n  Model OUTPUT at initialization:")
    print(f"    Recon range: [{recon_d.min():.4f}, {recon_d.max():.4f}]")
    print(f"    Recon RMS: mean={recon_rms.mean():.4f}")
    print(f"    Input RMS:  mean={audio_test[:16].squeeze(1).pow(2).mean(dim=1).sqrt().mean():.4f}")
    snr_init = compute_snr(audio_test[:16], recon_d)
    print(f"    SNR at init: {snr_init:.1f} dB" if snr_init else "    SNR at init: N/A")


# ============================================================
# TEST 8: CRITICAL — Multi-batch generalization test (no-FSQ vs FSQ)
# ============================================================
print("\n" + "=" * 70)
print("TEST 8: CRITICAL — Multi-batch generalization (50 different batches)")
print("=" * 70)

# Train on 50 different batches for 5 epochs each, test on unseen batch
# This simulates the full training scenario but faster

# A) Without FSQ
print("\n  A) Training WITHOUT FSQ on diverse data (50 batches x 5 epochs)...")
model_gen_a = SPPFAudioAutoencoder(512).to(device)
opt_gen_a = torch.optim.Adam(
    list(model_gen_a.encoder.parameters()) + list(model_gen_a.decoder.parameters()),
    lr=1e-3
)

for epoch in range(5):
    epoch_loss = 0
    for b in range(50):
        batch = make_fake_audio(32)
        z = model_gen_a.encoder(batch)
        recon = model_gen_a.decoder(z)
        loss = F.mse_loss(recon, batch)
        opt_gen_a.zero_grad()
        loss.backward()
        opt_gen_a.step()
        epoch_loss += loss.item()

    # Test on unseen
    with torch.no_grad():
        test_batch = make_fake_audio(64)
        z_test = model_gen_a.encoder(test_batch)
        recon_test = model_gen_a.decoder(z_test)
        test_mse = F.mse_loss(recon_test, test_batch).item()
        test_snr = compute_snr(test_batch, recon_test)
        snr_str = f"{test_snr:.1f}" if test_snr else "N/A"
    print(f"    Epoch {epoch+1}: train_mse={epoch_loss/50:.6f}, test_mse={test_mse:.6f}, test_SNR={snr_str} dB")

# B) With FSQ
print("\n  B) Training WITH FSQ on diverse data (50 batches x 5 epochs)...")
model_gen_b = SPPFAudioAutoencoder(512).to(device)
opt_gen_b = torch.optim.Adam(model_gen_b.parameters(), lr=1e-3)

for epoch in range(5):
    epoch_loss = 0
    for b in range(50):
        batch = make_fake_audio(32)
        recon, z_cont, z_q = model_gen_b(batch)
        loss = F.mse_loss(recon, batch)
        opt_gen_b.zero_grad()
        loss.backward()
        opt_gen_b.step()
        epoch_loss += loss.item()

    with torch.no_grad():
        test_batch = make_fake_audio(64)
        recon_test, _, _ = model_gen_b(test_batch)
        test_mse = F.mse_loss(recon_test, test_batch).item()
        test_snr = compute_snr(test_batch, recon_test)
        snr_str = f"{test_snr:.1f}" if test_snr else "N/A"
    print(f"    Epoch {epoch+1}: train_mse={epoch_loss/50:.6f}, test_mse={test_mse:.6f}, test_SNR={snr_str} dB")


# ============================================================
# TEST 9: Is decoder.final.clamp(-1, 1) killing gradients?
# ============================================================
print("\n" + "=" * 70)
print("TEST 9: Decoder output clamp analysis")
print("=" * 70)

with torch.no_grad():
    model_e = SPPFAudioAutoencoder(512).to(device)
    recon_e, _, _ = model_e(audio_test[:64])
    pre_clamp = model_e.decoder.final(
        model_e.decoder.block4(
            model_e.decoder.block3(
                model_e.decoder.block2(
                    model_e.decoder.block1(
                        model_e.encoder(audio_test[:64]).unsqueeze(-1)
                    )
                )
            )
        )
    )
    clamped = pre_clamp.clamp(-1, 1)
    pct_clamped = ((pre_clamp.abs() > 1.0).float().mean() * 100).item()
    print(f"  Pre-clamp range: [{pre_clamp.min():.4f}, {pre_clamp.max():.4f}]")
    print(f"  % of values clamped: {pct_clamped:.1f}%")
    print(f"  -> {'PROBLEM: Many values clamped = gradient killed!' if pct_clamped > 5 else 'OK: Few values clamped'}")


# ============================================================
# TEST 10: Commitment loss direction check
# ============================================================
print("\n" + "=" * 70)
print("TEST 10: Commitment loss analysis")
print("=" * 70)

with torch.no_grad():
    model_f = SPPFAudioAutoencoder(512).to(device)
    recon_f, z_cont_f, z_q_f = model_f(audio_test[:32])
    # In training: quant_loss = F.mse_loss(z_cont, z_q.detach())
    # This pushes z_cont toward z_q — OPPOSITE of what we want!
    # We want z_q to be close to z_cont (but z_q is discrete)
    # STE already handles gradients through z_q
    # The commitment loss should push encoder to produce values close to quantization points
    commit_loss = F.mse_loss(z_cont_f, z_q_f.detach())

    # How different are z_cont and z_q?
    diff = (z_cont_f - z_q_f).abs()
    print(f"  Commitment loss: {commit_loss.item():.6f}")
    print(f"  z_cont - z_q: mean_abs_diff={diff.mean():.4f}, max_diff={diff.max():.4f}")
    print(f"  z_cont stats: mean={z_cont_f.mean():.4f}, std={z_cont_f.std():.4f}")
    print(f"  z_q stats:    mean={z_q_f.mean():.4f}, std={z_q_f.std():.4f}")


# ============================================================
# TEST 11: Check if STFT loss dominates and fights MSE
# ============================================================
print("\n" + "=" * 70)
print("TEST 11: Loss landscape — do STFT and MSE agree on direction?")
print("=" * 70)

model_g = SPPFAudioAutoencoder(512).to(device)
stft_fn = MultiResolutionSTFTLoss(fft_sizes=(64, 128, 256)).to(device)
audio_11 = make_fake_audio(32)

# Compute gradients from MSE alone
recon_g, _, _ = model_g(audio_11)
mse_11 = F.mse_loss(recon_g, audio_11)
model_g.zero_grad()
mse_11.backward(retain_graph=True)
grad_mse = {n: p.grad.clone() for n, p in model_g.named_parameters() if p.grad is not None}

# Compute gradients from STFT alone
model_g.zero_grad()
stft_11 = stft_fn(recon_g, audio_11)
stft_11.backward(retain_graph=True)
grad_stft = {n: p.grad.clone() for n, p in model_g.named_parameters() if p.grad is not None}

# Check cosine similarity of gradients
cos_sims = []
for n in grad_mse:
    if n in grad_stft:
        g1 = grad_mse[n].flatten()
        g2 = grad_stft[n].flatten()
        cos = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
        cos_sims.append((n, cos))

# Show worst conflicting layers
cos_sims.sort(key=lambda x: x[1])
print(f"  Gradient cosine similarity (MSE vs STFT):")
print(f"  Most conflicting layers:")
for name, cos in cos_sims[:5]:
    print(f"    {name}: cos_sim={cos:.4f} {'CONFLICT!' if cos < 0 else 'aligned'}")
print(f"  Most aligned layers:")
for name, cos in cos_sims[-3:]:
    print(f"    {name}: cos_sim={cos:.4f}")
avg_cos = sum(c for _, c in cos_sims) / len(cos_sims)
print(f"  Average cosine similarity: {avg_cos:.4f}")
print(f"  -> {'PROBLEM: Losses fight each other!' if avg_cos < 0.3 else 'OK: Losses roughly agree'}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print("""
Review the numbers above to identify the single biggest bottleneck.
Key questions answered:
  1. Does FSQ destroy too much info? (Test 1-2)
  2. Can encoder+decoder reconstruct at all? (Test 3-4)
  3. Are gradients flowing through FSQ? (Test 5)
  4. Are loss weights balanced? (Test 6)
  5. Does the model generalize without FSQ? (Test 8)
  6. Is the clamp killing gradients? (Test 9)
  7. Do MSE and STFT fight each other? (Test 11)
""")
