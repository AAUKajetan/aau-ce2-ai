# DNN Class — Exam Overview

> Source modules: Module 2, Module 3, Module 9, Module 10  
> Exercises: Module 2, 3, 4 (Autoencoders/VAE), 5 (GANs/Transformers)

---

## Topics Map

| Topic | Key Files |
|-------|----------|
| Perceptron → MLP → Backprop | Module_2, Exercises_Module_2 |
| CNN (convolution, pooling, regularization) | Module_2, Exercises_Module_2 |
| RNN + BPTT + vanishing gradients | Module_3, Exercises_Module_3 |
| LSTM (gates, cell state) | Module_3, Exercises_Module_3 |
| GRU | Module_3 |
| Word embeddings | Module_3 |
| Autoencoders (basic, denoising, sparse, contractive) | Module_9, Exercises_Module_4 |
| VAE (ELBO, reparameterization trick) | Module_9, Exercises_Module_4 |
| GAN (minimax, training dynamics) | Module_10, Exercises_Module_5 |
| Attention mechanism (Bahdanau) | Module_10, Exercises_Module_5 |
| Transformer (self-attention, MHA, positional encoding) | Module_10, Exercises_Module_5 |

---

## 1. From Perceptron to MLP

| Model | Key Feature | Limitation |
|-------|------------|-----------|
| Perceptron | Binary classifier: sign(w·x + b) | Only linearly separable problems |
| Adaline | Linear activation + MSE | Still linear boundary |
| Logistic Regression | Sigmoid output, binary cross-entropy | Still linear boundary |
| MLP | Hidden layers + nonlinear activation | Vanishes with poor init/activations |

### MLP Forward Pass
For each layer l: `z^(l) = W^(l) a^(l-1) + b^(l)`, `a^(l) = f(z^(l))`

### Backpropagation (chain rule)
```
δ^(L) = ∇_a L ⊙ f'(z^(L))                          (output layer)
δ^(l) = (W^(l+1)ᵀ δ^(l+1)) ⊙ f'(z^(l))            (hidden layer)
∂L/∂W^(l) = δ^(l) (a^(l-1))ᵀ
W ← W − η ∇W L
```

### Activation Functions

| Function | Formula | Use |
|----------|---------|-----|
| Sigmoid | 1/(1+e^−z); σ'= σ(1−σ); max 0.25 | Binary output; vanishing gradient risk |
| Tanh | (e^z−e^−z)/(e^z+e^−z) | Hidden layers (zero-centered) |
| ReLU | max(0,z) | Default hidden layer choice |
| Leaky ReLU | max(αz, z) | Avoid dying ReLU |
| Softmax | e^zi / Σ e^zj | Multi-class output |

### Loss Functions
- **MSE**: regression
- **Binary cross-entropy**: binary classification
- **Categorical cross-entropy**: multi-class

---

## 2. Convolutional Neural Networks (CNNs)

### Why CNNs over MLPs for images?
MLPs: too many parameters, lose spatial structure. CNNs: local connectivity + parameter sharing + translation equivariance.

### Key Operations

**Convolution output size:** `⌊(W − k + 2p)/s⌋ + 1`
- W = input width, k = kernel size, p = padding, s = stride
- `padding="valid"` → no padding; `padding="same"` → output size = input size

**Pooling:** reduces spatial size, adds translation invariance. Max pooling most common.

**Typical block:** `Conv → BatchNorm → ReLU → Pooling`

**Architecture pattern:**
```
[Conv + ReLU + Pool] × N → Flatten → FC layers → Softmax
```
Early layers: edges/textures → deeper layers: shapes/objects.

### Regularization Techniques

| Technique | Mechanism |
|-----------|-----------|
| L1/L2 regularization | Penalize large weights in loss |
| Dropout | Randomly zero activations during training |
| Batch Normalization | Normalize layer inputs; stabilizes training |
| Data augmentation | Flip, crop, rotate, color jitter |
| Early stopping | Stop when validation loss stops improving |

---

## 3. RNNs and the Vanishing Gradient Problem

### Why RNNs for sequential data?
MLPs/CNNs: fixed-size inputs, no memory. RNNs: hidden state hₜ carries information across time.

### Vanilla RNN Forward Pass
```
hₜ = tanh(Wₓ xₜ + Wₕ hₜ₋₁ + b)
yₜ = Wᵧ hₜ + bᵧ
```
Same weights at every step → parameter sharing over time.

### Backpropagation Through Time (BPTT)
Unroll RNN for T steps, apply backprop. Gradient of hₜ w.r.t. h₁ multiplied by Whh at each step:
- **Vanishing**: |Whh| < 1 → gradients → 0 → network forgets distant past
- **Exploding**: |Whh| > 1 → gradients → ∞ → fix with gradient clipping

### RNN Variants by Shape

| Type | Description | Example |
|------|-------------|---------|
| Many-to-one | Sequence → single output | Sentiment analysis |
| Many-to-many (sync) | Same length in/out | POS tagging |
| Many-to-many (async) | Encoder-decoder | Machine translation |

---

## 4. LSTM

Designed to solve vanishing gradients via a **cell state** Cₜ with additive updates.

### Gates

| Gate | Formula | Purpose |
|------|---------|---------|
| Forget fₜ | σ(Wf [hₜ₋₁, xₜ] + bf) | What to erase from cell state |
| Input iₜ | σ(Wi [hₜ₋₁, xₜ] + bi) | What new info to write |
| Cell candidate C̃ₜ | tanh(Wc [hₜ₋₁, xₜ] + bc) | Candidate values |
| Output oₜ | σ(Wo [hₜ₋₁, xₜ] + bo) | What to expose as hₜ |

### Update Equations
```
Cₜ = fₜ ⊙ Cₜ₋₁  +  iₜ ⊙ C̃ₜ     ← additive update preserves gradients
hₜ = oₜ ⊙ tanh(Cₜ)
```

### GRU (simplified LSTM)
Two gates (reset rₜ, update zₜ). Fewer parameters; comparable performance to LSTM.

### RNN vs LSTM

| | RNN | LSTM |
|--|-----|------|
| Memory | Hidden state only | Hidden state + cell state |
| Long-term deps | Poor (vanishing grads) | Good (additive cell update) |
| Complexity | Low | Higher (4× parameters) |

---

## 5. Word Embeddings

- Map discrete tokens → dense real-valued vectors
- **Word2Vec** (Skip-gram / CBOW): predict context from word, or word from context
- Embeddings encode semantic similarity: king − man + woman ≈ queen
- Pre-trained (GloVe, fastText): transfer knowledge to downstream tasks

---

## 6. Autoencoders

### Basic Autoencoder
```
Input x → Encoder fθ → Latent z → Decoder gφ → Reconstruction x̂
```
Loss: `L = ||x − x̂||²` (MSE) or binary cross-entropy.

The bottleneck forces the encoder to discard noise and keep only essential structure.

**Limitation**: deterministic; latent space unstructured → cannot meaningfully sample new points.

### Regularized Variants

| Model | Extra | Key Property |
|-------|-------|-------------|
| Denoising AE | Noisy input x̃; reconstruct clean x | Robust, noise-invariant features |
| Sparse AE | + λ \|\|z\|\|₁ penalty | Few active neurons |
| Contractive AE | + λ \|\|Jfθ(x)\|\|²F penalty | Smooth, stable latent space |

**Latent too large** → model can memorize (identity mapping); no compression; no generalization.

---

## 7. Variational Autoencoder (VAE)

### Motivation
Make latent space a proper probability distribution → enable sampling of new, meaningful data.

### Architecture
- Prior: z ~ N(0, I)
- Encoder (inference network): outputs **μ(x)** and **log σ²(x)** instead of a point z
- Decoder: P(x|z)

### Reparameterization Trick
Standard sampling z ~ N(μ, σ²) is non-differentiable. Rewrite:
```
ε ~ N(0, I)
z = μ(x) + σ(x) ⊙ ε
```
Gradients flow through μ and σ; ε treated as fixed noise.

### VAE Loss (ELBO)
```
L = E_q[log P(x|z)]  −  KL(q(z|x) || N(0,I))
```
- **Reconstruction term**: how well decoder reconstructs x
- **KL term**: regularizes latent space toward N(0,I); enforces continuity and completeness

KL closed form for Gaussians:
```
KL = -½ Σ (1 + log σᵢ² − μᵢ² − σᵢ²)
```

### β-VAE
`L = E[log P(x|z)] − β · KL(...)` — β > 1 → more disentangled latent factors; trade-off: reconstruction quality.

### AE vs VAE

| | AE | VAE |
|--|----|-----|
| Latent space | Unstructured point | Gaussian distribution |
| Sampling | No | Yes (reparameterization) |
| Loss | Reconstruction only | Reconstruction + KL |

---

## 8. GANs (Generative Adversarial Networks)

### Core Idea
Two networks compete:
- **Generator G(z)**: maps noise z ~ P(z) → fake data
- **Discriminator D(x)**: probability that x is real

### Minimax Objective
```
min_G max_D  E_x[log D(x)]  +  E_z[log(1 − D(G(z)))]
```

### Training Loop
```
# Step 1: Train D (freeze G)
d_loss = BCE(D(real), 1) + BCE(D(fake.detach()), 0)

# Step 2: Train G (freeze D) — non-saturating form
g_loss = BCE(D(G(z)), 1)   # maximize log D(G(z)) instead of minimizing log(1-D(G(z)))
```

### Why Non-Saturating Loss for G?
Early training: G generates noise → D(G(z)) ≈ 0 → gradient of log(1−D(G(z))) → 0 → G gets no learning signal. Fix: train G to **maximize** log D(G(z)).

### Common Failure Modes

| Problem | Description | Fix |
|---------|-------------|-----|
| Mode collapse | G produces limited variety | Feature matching, mini-batch discrimination |
| Non-convergence | G and D oscillate | Careful hypertuning, WGAN |
| D too strong | No gradient to G | Balance training steps |

### Nash Equilibrium
Theoretical optimum: G reproduces real data distribution → D(x) = 0.5 everywhere.

---

## 9. Attention and Transformers

### Attention Mechanism (Bahdanau 2015)
Encoder produces hidden states (h₁,...,hT). Decoder at step t computes:
```
cₜ = Σⱼ αₜⱼ hⱼ
αₜⱼ = softmax(eₜⱼ),   eₜⱼ = score(sₜ₋₁, hⱼ)
```
Allows decoder to focus on relevant encoder positions; no information bottleneck.

### Scaled Dot-Product Attention
```
Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) · V
```
- `/ √dₖ`: prevents dot products from growing large → keeps softmax gradients well-behaved
- n² scalar dot products per head for sequence length n

### Multi-Head Attention
```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ) Wₒ
headᵢ = Attention(Q Wᵢᴼ, K Wᵢᴷ, V Wᵢᵛ)
```
Each head can capture different types of relationships. For h heads: h·n² dot products total.

### Positional Encoding
Self-attention is permutation-invariant → add sinusoidal encoding to inject order:
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

### Transformer Encoder Block
```
x → Multi-Head Self-Attention → Add & Norm → Feed-Forward → Add & Norm → output
```
- **Residual connections**: prevent vanishing gradients, preserve information flow
- **Layer normalization**: stabilizes activations
- **Masked self-attention** (decoder): causal — attends only to previous positions

### Transformer vs RNN

| | RNN + Attention | Transformer Self-Attention |
|--|----------------|--------------------------|
| Computation | Sequential | Parallel |
| Long-range deps | Limited by hidden state | O(1) path length |
| Memory | O(n) hidden states | O(n²) attention matrix |
| Scalability | Poor | Excellent |

---

## 10. Exam Question Patterns (from exercises)

**MLP/Backprop:**
- Derive δ^(L) and δ^(l) from chain rule
- Why does sigmoid contribute to vanishing gradients? (max gradient 0.25)
- Effect of more neurons / more data on MNIST accuracy

**CNN:**
- Compute output size given W, k, p, s
- Why fewer parameters than FC? (weight sharing)
- padding="valid" vs padding="same"
- Why pooling? (invariance, dimension reduction)

**RNN/LSTM:**
- Why are FFNs unsuitable for sequential data?
- Explain vanishing gradient in BPTT
- Name all 4 LSTM gates and their purpose
- Why does LSTM's cell state help with vanishing gradients? (additive update)

**AE/VAE:**
- Reconstruction loss: when to use MSE vs BCE?
- What happens if latent dimension is too large?
- KL divergence role in VAE — what does it enforce?
- Write the reparameterization trick — why is it needed?
- Denoising AE: what corruption is applied and why?

**GAN:**
- Write and explain the minimax objective
- Why does G's gradient vanish early in training? How to fix?
- Why use BatchNorm in G, LeakyReLU in D?
- What is mode collapse?

**Transformer:**
- Resolve coreference with attention ("it" = animal, not street)
- Count dot products: n=4, h=2 heads → 32 dot products
- Implement scaled dot-product attention from scratch
- Why scale by √dₖ?
- What is masked self-attention used for?

---

## 11. Abbreviations

| Abbr | Meaning | Topic |
|------|---------|-------|
| MLP | Multi-Layer Perceptron | §1 |
| FC / FFN | Fully Connected / Feed-Forward Network | §1 |
| MSE | Mean Squared Error | §1 |
| CE / BCE | (Binary) Cross-Entropy | §1 |
| ReLU | Rectified Linear Unit | §1 |
| CNN | Convolutional Neural Network | §2 |
| BN | Batch Normalization | §2 |
| RNN | Recurrent Neural Network | §3 |
| BPTT | Backpropagation Through Time | §3 |
| POS | Part of Speech (tagging) | §3 |
| LSTM | Long Short-Term Memory | §4 |
| GRU | Gated Recurrent Unit | §4 |
| CBOW | Continuous Bag of Words | §5 |
| AE | Autoencoder | §6 |
| VAE | Variational Autoencoder | §7 |
| ELBO | Evidence Lower Bound | §7 |
| KL | Kullback–Leibler divergence | §7 |
| GAN | Generative Adversarial Network | §8 |
| WGAN | Wasserstein GAN | §8 |
| MHA | Multi-Head Attention | §9 |
| Q / K / V | Query / Key / Value | §9 |
| PE | Positional Encoding | §9 |
