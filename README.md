# Stable Diffusion from Scratch

A complete implementation of a text-to-image diffusion model built from scratch in PyTorch, including a custom VAE encoder/decoder, U-Net with cross-attention, and DDPM sampling.

## Overview

This project implements a latent diffusion model similar to Stable Diffusion, trained on the COCO dataset. All components are built from PyTorch primitives without using pre-trained diffusion models or high-level libraries.

### Generated Samples

**Prompt: "a dog on the grass"**
- Model generates animal-like shapes with appropriate outdoor colors (greens, browns).
- Shows foreground objects against background scenery.

**Prompt: "a red sports car"**
- Model generates vehicle-like shapes with prominent red regions.
- Demonstrates basic text conditioning and prompt-following behavior.

## Architecture

### 1. VAE (Variational Autoencoder)

- Architecture: 4-block encoder/decoder with block channels `[64, 128, 256, 256]`.
- Latent channels: 4.
- Compression: 128×128 images → 16×16 latent tensors (8× spatial downsampling).
- Objective: MSE reconstruction loss + β·KL divergence with β = 0.00001.
- Resulting latent distribution after training:
  - Mean ≈ 0.0.
  - Standard deviation ≈ 0.9.

### 2. U-Net (Latent Diffusion Model)

- Parameters: ~21M.
- Operates in VAE latent space (shape `4 × 16 × 16`).
- Uses:
  - Sinusoidal time embeddings + MLP.
  - Cross-attention with CLIP text embeddings.
  - Self-attention on images at lowest resolution
  - ResNet-style blocks with GroupNorm.
  - U-shaped encoder–decoder with skip connections.

### 3. Text Encoder

- Pre-trained CLIP text encoder (frozen).
- Produces 512-d embeddings for input prompts.
- Text embeddings are injected via cross-attention in the U-Net.

### 4. Diffusion Process

- Timesteps: 200.
- Beta schedule: linear from 1e-4 to 0.02.
- Forward process:
  - Adds noise according to cumulative product of alphas (ᾱ_t).
- Reverse process:
  - U-Net predicts noise ε and denoises using the DDPM update rule.

## Training Details

### VAE Training

- Dataset: COCO 2017 (subset of 40k images).
- Image size: 128×128.
- Batch size: 32.
- Epochs: 15.
- Optimizer: Adam, learning rate 1e-4.
- β (KL weight): 0.00001.
- Hardware: Apple M4 (Metal backend).
- Approximate training time: ~10 hours.

Final VAE metrics (on training batches):

- Latent mean: about -0.001.
- Latent std: about 0.908.
- Reconstruction MSE: ≈ 0.023.
- Visual reconstructions: blurry but preserve global structure, colors, and layout.

### U-Net / Diffusion Training

- Dataset: COCO 2017 (same 40k images).
- Latents: obtained by encoding images with the trained VAE (no extra scaling).
- Batch size: 32.
- Epochs: 15.
- Optimizer: Adam with weight decay 1e-4.
- Learning rate: 1e-4 with cosine annealing LR scheduler.
- Timesteps: 200.
- Loss: mean squared error between predicted noise and true noise.
- EMA: Exponential moving average U-Net maintained for experiments (high decay proved suboptimal; base U-Net used for best samples).
- Hardware: Apple M4.
- Approximate training time: ~12 hours.

Final diffusion metric:

- Training loss after 15 epochs: ≈ 0.554.

## Key Challenges and Solutions

### 1. VAE Latent Distribution Mismatch

**Problem**

- Initial VAE used a very small KL weight (β = 0.000001), leading to:
  - Latent std ≈ 0.35.
  - Distribution far from the standard normal prior p(z) = N(0, 1).
- Diffusion model assumes latents are drawn from N(0, 1) at the start of the reverse process, so this mismatch prevented learning.

**Solution**

- Increased β to 0.00001 (10× larger).
- Deepened the VAE to `[64, 128, 256, 256]` blocks to maintain reconstruction quality under stronger regularization.
- After retraining, latent statistics became:
  - Mean ≈ 0.
  - Std ≈ 0.9.
- This made the VAE approximate posterior q(z|x) compatible with the diffusion prior, enabling successful U-Net training.

**ELBO Perspective**

The VAE maximizes the Evidence Lower Bound (ELBO):

ELBO = E_{q(z|x)}[log p(x|z)] − β · KL(q(z|x) || p(z))

- First term: reconstruction quality (implemented via MSE).
- Second term: encourages q(z|x) to match N(0, 1).
- β controls the trade-off; too small ignores KL, too large over-regularizes.

### 2. EMA Model Degrading Sample Quality

**Problem**

- EMA U-Net used decay = 0.9999.
- With relatively few epochs, EMA updated too slowly and remained close to initial weights.
- EMA-based samples looked worse and more chaotic than samples from the base U-Net.

**Finding**

- For this training regime and model size, the base U-Net (non-EMA) produced much better and more structured images.
- EMA with such a high decay is better suited for very long training runs or larger models.

**Outcome**

- For final sampling, the regular U-Net is preferred.
- EMA is kept only for experimentation; decay can be lowered (e.g., 0.995) in future work.

### 3. Scaling Factor Misuse

**Problem**

- Stable Diffusion uses a scaling factor (≈0.18215) because its VAE latents have a much larger std (~5.5).
- This project initially copied that factor even though the custom VAE’s latents had std ~0.35 (and later ~0.9).
- Extra scaling made latents too small and harmed learning.

**Solution**

- Removed manual scaling entirely:
  - During training: `z = mu + sigma * torch.randn_like(sigma)`.
  - During sampling: directly decode `vae.decoder(z_t)`.
- Since the VAE already produces approximately unit-variance latents, no extra scaling is needed.

## Project Structure

├── vae.py          # VAE encoder/decoder implementation
├── train_vae.py    # VAE training script
├── unet.py         # U-Net architecture with time & text conditioning
├── train_unet.py   # Diffusion model training script
├── sample.py       # Example inference / sampling script (optional)
└── README.md



## Requirements

torch>=2.0.0
torchvision
Pillow
matplotlib
transformers # for CLIP text encoder
numpy



Install via:

pip install torch torchvision pillow matplotlib transformers numpy



## Usage

### 1. Train the VAE

python train_vae.py


This trains the AutoencoderKL on COCO images and saves a checkpoint, e.g.:

- `vae_ckpt_epoch15.pth`

### 2. Train the Diffusion U-Net

python train_unet.py


This:

- Loads the trained VAE.
- Encodes images to latents.
- Trains the U-Net to predict noise for 200 diffusion steps.
- Saves checkpoints such as:
  - `unet_ckpt15.pth`
  - `t_ember_ckpt15.pth` (time embedding)
  - Optional EMA checkpoints.

### 3. Generate Images

Example (inside `train_unet.py` or a separate `sample.py`):

Load config and devices
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Load VAE
``` python
vae = AutoencoderKL(...)
vae.load_state_dict(torch.load("vae_ckpt_epoch15.pth", map_location=device))
vae.to(device).eval()

Load U-Net and time embedding
cfg = SDConfig(...) # same config as during training
unet = UNetLatent(cfg).to(device)
unet.load_state_dict(torch.load("unet_ckpt15.pth", map_location=device))
t_ember = TimeEmbedding(cfg.time_dim).to(device)
t_ember.load_state_dict(torch.load("t_ember_ckpt15.pth", map_location=device))
```

# Use CLIP text encoder (frozen)
``` python
text_encoder = ClipTextEncoder().to(device).eval()

Sampling function (simplified)
def sample(unet, vae, t_embedder, text_encoder, prompt, num_steps=200, num_samples=4):
unet.eval()
with torch.no_grad():
# Encode text
prompts = [prompt] * num_samples
text_emb = text_encoder(prompts)


# Start from Gaussian noise in latent space
z = torch.randn(num_samples, cfg.latent_channels, 16, 16, device=device)

# Precompute schedule
beta, alpha, alpha_bar = baab()
beta, alpha, alpha_bar = beta.to(device), alpha.to(device), alpha_bar.to(device)

for t in reversed(range(num_steps)):
    t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
    t_emb = t_embedder(t_batch)
    eps_pred = unet(z, t_emb, text_emb)

    alpha_t = alpha[t]
    alpha_bar_t = alpha_bar[t]
    beta_t = beta[t]

    if t > 0:
        noise = torch.randn_like(z)
    else:
        noise = torch.zeros_like(z)

    z = (1.0 / torch.sqrt(alpha_t)) * (
            z - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred
        ) + torch.sqrt(beta_t) * noise

# Decode latents to images
x = vae.decoder(z).clamp(-1, 1)
x = (x + 1) / 2  # to  for visualization[1]
return x
```

Then save or visualize the generated grid with `torchvision.utils.make_grid`.

## Results and Analysis

### What Works

- Text conditioning has a clear effect:
  - Prompts about dogs produce animal-like shapes on natural-looking backgrounds.
  - Prompts about red cars produce vehicle-like shapes with red highlights.
- The model:
  - Distinguishes coarse categories (animals vs vehicles).
  - Produces plausible color palettes and foreground/background separation.
  - Learns nontrivial structure from scratch.

### Limitations

- Images are blurry due to VAE reconstruction quality and 128×128 resolution.
- Fine details (faces, textures, small objects) are not captured.
- EMA configuration was not fully optimized in this run.
- Training is performed on only 40k COCO images, not the full dataset.

### Future Improvements

1. **Sharper VAE**  
   - Experiment with slightly smaller β (e.g., 0.000005) to allow sharper reconstructions while still keeping latent std in a usable range.

2. **Higher Resolution**  
   - Extend the architecture to 256×256 or 512×512 with additional down/up-sampling blocks.

3. **More Data**  
   - Train on the full COCO 2017 dataset (~118k images) or other large-scale datasets.

4. **Classifier-Free Guidance**  
   - Implement CFG by training with conditional + unconditional embeddings to improve prompt alignment and sample quality.

5. **Improved Sampling**  
   - Add DDIM or other samplers for faster and sometimes better-quality generation.

## Lessons Learned

- Getting the **VAE latent distribution** right (mean ≈ 0, std ≈ 1) is critical for diffusion models that assume a standard normal prior.
- **β in the VAE loss** is not a cosmetic hyperparameter; it directly controls how compatible the latent space is with diffusion.
- **EMA decay** must be tuned to the training regime; very high decay can lock the EMA model near initialization for many epochs.
- Copying **scaling factors** from other implementations can break training if your VAE behaves differently; always measure your own latent stats.
- Blurry outputs do not necessarily mean the diffusion model failed; sometimes the limitation is in the decoder, not the denoiser.

## Acknowledgments

- Stable Diffusion and latent diffusion research for architectural inspiration.
- COCO dataset for providing image–caption pairs.
- CLIP for robust text embeddings used for conditioning.

## License

MIT

---

This project is intended as an educational, from-scratch exploration of latent diffusion models, not as a production-ready image generator.