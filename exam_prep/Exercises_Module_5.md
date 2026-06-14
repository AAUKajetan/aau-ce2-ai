# Exercises — Module 5 (Generative Modeling II): GANs and Attention/Transformers

## GAN Questions

1. **GAN minimax objective:**
   ```
   min_G max_D E_x[log D(x)] + E_z[log(1-D(G(z)))]
   ```
   - `E_x[log D(x)]`: D correctly identifies real samples
   - `E_z[log(1-D(G(z)))]`: D correctly rejects fake samples
   - Formulated as a game: D tries to discriminate, G tries to fool D — adversarial training
   - If D becomes too strong too quickly: D(G(z))→0, `log(1-D(G(z)))` saturates → gradient signal to G vanishes → G can't learn. Fix: train G with `max log D(G(z))` instead.

2. **GAN vanishing gradients early in training:**
   - Initially G generates noise → D easily rejects → D(G(z)) ≈ 0
   - Gradient of `log(1-D(G(z)))` → 0 when D(G(z))→0
   - Fix: use non-saturating loss for G: maximize `log D(G(z))` instead of minimizing `log(1-D(G(z)))`

3. **GAN for 28×28 MNIST:**
   - **Generator:** Linear(z_dim→256)→BN→LeakyReLU → Linear(256→512)→BN→LeakyReLU → Linear(512→784)→Tanh, reshape to (1,28,28)
   - **Discriminator:** Flatten → Linear(784→512)→LeakyReLU → Linear(512→256)→LeakyReLU → Linear(256→1)→Sigmoid
   - **BatchNorm:** stabilizes training, reduces internal covariate shift in G
   - **LeakyReLU in D:** prevents dead neurons (standard ReLU can kill gradients in D)

4. **GAN training loop:**
   ```python
   for real_batch in dataloader:
       z = torch.randn(batch_size, z_dim)
       fake = G(z)
       
       # Train D
       d_loss = BCE(D(real_batch), ones) + BCE(D(fake.detach()), zeros)
       d_optim.zero_grad(); d_loss.backward(); d_optim.step()
       
       # Train G
       g_loss = BCE(D(fake), ones)  # non-saturating form
       g_optim.zero_grad(); g_loss.backward(); g_optim.step()
   ```
   - Sample z ~ N(0,I) for each G update
   - Detach fake when training D to prevent gradients flowing into G

---

## Attention / Transformer Questions

5. **"The animal didn't cross the street because it was too tired" — what does "it" refer to?**
   - "it" refers to the animal (tired animal can't cross)
   - Self-attention: each token computes attention weights over all other tokens
   - The query for "it" attends most strongly to "animal" (high dot product similarity)
   - RNNs would need to propagate this information through many timesteps; attention resolves it directly

6. **Scalar dot products per head in attention matrix:**
   - For sequence length n and 1 head: n² dot products (each of n queries attends to n keys)
   - For h heads: h·n² scalar dot products total
   - Example: n=4, 1 head → 16 dot products; with h=2 heads → 32

7. **Scaled dot-product attention from scratch (numpy):**
   ```python
   def attention(Q, K, V):
       d_k = Q.shape[-1]
       scores = Q @ K.T / np.sqrt(d_k)       # (4,4)
       weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
       weights /= weights.sum(axis=-1, keepdims=True)   # softmax, rows sum to 1
       return weights @ V
   
   # Test with Q=K=V of shape (4,8): d_k=8
   Q = np.random.randn(4, 8)
   K = np.random.randn(4, 8)
   V = np.random.randn(4, 8)
   out = attention(Q, K, V)  # shape (4,8)
   # Verify: weights.sum(axis=-1) should be all 1.0
   ```
   - `/ √d_k` prevents dot products from growing large for high-dimensional keys, keeping softmax gradients well-behaved

8. **Multi-Head Attention:**
   ```python
   def multi_head_attention(Q, K, V, h=2):
       d_model = Q.shape[-1]
       d_k = d_model // h
       heads = []
       for i in range(h):
           # Project to subspace
           Qi = Q @ Wq[i]   # (n, d_k)
           Ki = K @ Wk[i]
           Vi = V @ Wv[i]
           heads.append(attention(Qi, Ki, Vi))
       concat = np.concatenate(heads, axis=-1)   # (n, d_model)
       return concat @ Wo   # final projection
   ```
   - Each head can focus on different types of relationships (local syntax, long-range coreference, etc.)
   - Outputs concatenated and projected back to d_model
