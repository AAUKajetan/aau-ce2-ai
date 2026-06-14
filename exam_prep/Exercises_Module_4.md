# Exercises — Module 4 (Generative Modeling I): Autoencoders and VAEs

## Theory Questions

1. **Autoencoder structure:**
   - Encoder: compresses input x → latent z (bottleneck)
   - Latent space z: compressed representation
   - Decoder: reconstructs x from z
   - Bottleneck forces the model to learn essential features; discards noise

2. **Reconstruction loss:**
   - MSE: `L = ||x - x̂||²` — good for continuous data (images as floats)
   - Binary Cross-Entropy: `L = -Σ[x·log x̂ + (1-x)·log(1-x̂)]` — for binary/[0,1] pixel values (sigmoid output)
   - Choose MSE when outputs are continuous; BCE when treating pixels as Bernoulli probabilities

3. **Implement FC autoencoder for MNIST (28×28→32→28×28):**
   ```python
   class AE(nn.Module):
       def __init__(self):
           self.encoder = nn.Sequential(
               nn.Flatten(),
               nn.Linear(784, 256), nn.ReLU(),
               nn.Linear(256, 32)
           )
           self.decoder = nn.Sequential(
               nn.Linear(32, 256), nn.ReLU(),
               nn.Linear(256, 784), nn.Sigmoid(),
               nn.Unflatten(1, (1, 28, 28))
           )
   ```
   Loss: MSELoss() between output and original input

4. **Latent dimension too large:**
   - Can memorize individual training samples (identity mapping)
   - No compression → no forced generalization
   - Unlike supervised learning where generalization is measured on held-out labels; here the AE can reconstruct training data perfectly without learning structure

5. **Denoising autoencoder:**
   - Add Gaussian noise: `x_noisy = x + σ·randn_like(x)`, clip to [0,1]
   - Target: reconstruct clean x from noisy x_noisy
   - Evaluate: compare MSE of clean reconstruction vs noisy input
   - Forces encoder to learn robust features invariant to noise

6. **VAE vs traditional AE:**
   - AE: deterministic mapping x → z → x̂; latent space unstructured
   - VAE: probabilistic latent space z ~ N(μ(x), σ²(x)); encoder outputs distribution parameters, not a point; enables sampling of new data points

7. **KL divergence in VAE:**
   - KL(q(z|x) || N(0,I)) regularizes latent space toward standard normal
   - Enforces: continuity (nearby z → similar outputs) and completeness (any sampled z decodes to plausible output)
   - Without KL: AE degenerates — σ→0, becomes deterministic AE with no generative capability

8. **Reparameterization trick:**
   ```python
   # Non-differentiable (can't backprop through sampling):
   z ~ N(μ, σ²)
   
   # Reparameterized (differentiable):
   ε ~ N(0, I)
   z = μ + σ * ε
   ```
   Gradients can flow through μ and σ; ε is treated as fixed noise. Required for backpropagation.

9. **Full VAE on MNIST:**
   ```python
   # Encoder outputs μ and log_σ²
   μ, log_var = encoder(x)
   σ = torch.exp(0.5 * log_var)
   
   # Reparameterize
   ε = torch.randn_like(σ)
   z = μ + σ * ε
   
   # Decode
   x_hat = decoder(z)
   
   # Loss = reconstruction + KL
   recon_loss = F.mse_loss(x_hat, x)
   kl_loss = -0.5 * torch.sum(1 + log_var - μ**2 - torch.exp(log_var))
   loss = recon_loss + kl_loss
   ```
   Generate: sample z ~ N(0,I) and decode

10. **Latent space visualization and interpolation:**
    - Reduce to 2D using 2D latent space or t-SNE
    - Plot: scatter of encoded test samples colored by digit class → should show clusters
    - Interpolation: take z₁ and z₂ from two samples, generate z_interp = α·z₁ + (1-α)·z₂ for α ∈ [0,1], decode each → smooth transition between digits
