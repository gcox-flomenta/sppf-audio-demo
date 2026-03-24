# Lessons from φ-Diffusion — Apply to Audio Codec

Critical fixes discovered during φ-Diffusion (video/image) training that directly apply to the SPPF audio codec.

---

## 1. CRITICAL: Never Clamp KL Loss

**Bug we found:** Using `loss_kl = 50.0 + 0.01 * softplus(kl_raw - 50.0)` outputs a constant ~50 regardless of actual KL. The encoder receives ZERO gradient from KL. It never learns to produce Gaussian latents.

**Fix:** Use raw KL directly with no clamp:
```python
# WRONG (zero gradient):
loss_kl = torch.clamp(kl_raw, max=50.0)  # hard clamp — zero gradient above 50
loss_kl = 50.0 + 0.01 * F.softplus(kl_raw - 50.0)  # always ≈50

# CORRECT:
loss_kl = kl_raw  # direct, full gradient always
```

If KL explodes, fix the source (logvar clamp, spectral norm) — don't hide it with loss clamping.

---

## 2. Spectral Norm on Encoder FC Layers

Prevents encoder weights from exploding, which causes KL spikes to millions.

```python
# In your audio encoder:
self.fc_mean = nn.utils.spectral_norm(nn.Linear(hidden_dim, latent_dim))
self.fc_logvar = nn.utils.spectral_norm(nn.Linear(hidden_dim, latent_dim))
```

Without spectral norm, outlier audio samples (loud transients, unusual frequencies) cause logvar to spike, producing KL in the millions for that batch.

---

## 3. Tight logvar Clamp

```python
# In encoder forward():
logvar = self.fc_logvar(h).clamp(-6.0, 6.0)
```

- `logvar=6` → std=exp(3)=20, max per-dim KL≈200 (manageable)
- `logvar=20` (Stable Diffusion default) → std=exp(10)=22026 → KL=millions (explosion)

For a 64-dim audio latent, logvar clamp [-6, 6] bounds total KL to ~12,800 max. Plenty of room for the encoder to learn.

---

## 4. φ-Scaled KL (Novel — Better Than Standard)

Instead of forcing all latent dims to N(0,1), each dim has φ-scaled target variance:

```python
PHI_INV = 0.6180339887498949

def phi_kl_loss(mean, logvar, latent_dim=64):
    """φ-Scaled KL for audio latent."""
    scale = 14.0 / latent_dim  # smooth decay across dims
    indices = torch.arange(latent_dim, device=mean.device, dtype=torch.float32)
    phi_var = PHI_INV ** (indices * scale)

    var_q = logvar.exp()
    kl_per_dim = 0.5 * (var_q / phi_var + mean.pow(2) / phi_var - 1 - logvar + phi_var.log())
    return kl_per_dim.mean()
```

For 64-dim audio latent:
- Dim 0 (var=1.0): coarse features — pitch, energy, voiced/unvoiced
- Dim 32 (var≈0.03): mid features — formant structure, spectral envelope
- Dim 63 (var≈0.001): fine features — breathiness, micro-texture

This matches how audio naturally decomposes: pitch varies more than texture.

---

## 5. Fixed Beta = 0.001 (Path C)

Don't use adaptive beta controllers, don't use annealing schedules. Just:

```python
beta = 0.001
loss = mse_loss + vgg_weight * vgg_loss + beta * kl_loss
```

- `beta=0.001`: strong enough for approximate Gaussian (instant generation on cheap device)
- Combined with spectral norm + logvar clamp: stable training, no explosions
- This is what production models converge to after trying everything else

---

## 6. FSQ Instead of RVQ

If using RVQ (Residual Vector Quantization) for the audio codebook, watch for:
- Codebook collapse (dead codes — 10-30% typical)
- Training instability
- Complex multi-stage residual training

FSQ (Finite Scalar Quantization) eliminates all of these:
```python
# From DESE model repo: C:\code\DESE model\src\model\abstraction.py
# Class: FSQCodebook

# FSQ in SEM achieved 97.6% codebook utilization (590/625 codes active)
# RVQ typically achieves 70-90%
```

For audio at 64-dim latent:
- FSQ with 8 dims × 5 levels = 390,625 codes → 24 bits → 1.2 kbps at 50fps
- FSQ with 16 dims × 5 levels = billions of codes → 48 bits → 2.4 kbps at 50fps

---

## 7. Golden Ratio Quantization (GRQ)

For transmitting audio latents over the wire:

```python
# From DESE model repo: C:\code\DESE model\src\compression\golden_ratio_quant.py
# Class: GoldenRatioQuantizer

quantizer = GoldenRatioQuantizer(n_bits=8)
indices, scale = quantizer.quantize(latent)  # 64 bytes per frame
latent_reconstructed = quantizer.dequantize(indices, scale)
```

φ-spaced bucket boundaries are optimal for Laplace-distributed neural activations. Better reconstruction than uniform quantization at the same bit budget.

---

## 8. DESE in Latent Space

Compare consecutive audio latent vectors for ONSET/CHANGE/SILENCE:

```python
def classify_audio_frame(z_current, z_baseline, onset_thresh=0.5, change_thresh=0.1):
    diff = torch.norm(z_current - z_baseline).item()

    if z_baseline is None:
        return "ONSET"  # first frame
    elif diff > onset_thresh:
        return "ONSET"  # major change (new speaker, scene cut)
    elif diff > change_thresh:
        return "CHANGE"  # speech continues
    elif diff < 0.01:
        return "SILENCE"  # no audio
    else:
        return "STEADY"  # similar to last (sustained vowel)
```

SILENCE frames: 0 bytes transmitted.
STEADY frames: 0 bytes (receiver holds last decoded frame).
CHANGE frames: send latent diff (~32 bytes instead of 64).
ONSET frames: send full latent (64 bytes).

Average call at 50fps: ~1-3 kbps instead of 25+ kbps.

---

## 9. Training Recipe Summary

```python
# Audio VAE training — proven stable configuration
encoder = AudioEncoder(latent_dim=64)
# Apply spectral norm to FC layers
encoder.fc_mean = nn.utils.spectral_norm(encoder.fc_mean)
encoder.fc_logvar = nn.utils.spectral_norm(encoder.fc_logvar)

# In encoder forward:
logvar = self.fc_logvar(h).clamp(-6.0, 6.0)  # tight clamp

# Loss:
loss_mse = F.mse_loss(x_rec, x)
loss_stft = multi_resolution_stft_loss(x_rec, x)  # perceptual for audio
loss_kl = phi_kl_loss(mean, logvar)  # φ-scaled, NOT standard KL

beta = 0.001  # fixed, no annealing
loss = loss_mse + 0.1 * loss_stft + beta * loss_kl
# NO CLAMP on kl. Ever.
```

---

## Files to Import

```python
import sys
sys.path.insert(0, r"C:\code\DESE model")

# Golden Ratio Quantization
from src.compression.golden_ratio_quant import GoldenRatioQuantizer

# FSQ (if replacing RVQ)
from src.model.abstraction import FSQCodebook
```

---

*Lessons from φ-Diffusion training, March 2026. Apply to SPPF audio codec to avoid the same mistakes.*
