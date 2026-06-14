# Exercises — Module 3: RNN and LSTM

## Theory Questions

1. **Why are FFNs unsuitable for sequential data?**
   - Fixed input size, no memory of previous inputs, can't handle variable-length sequences
   - RNNs address this via hidden state hₜ = f(Wxhxₜ + Whhh_{t-1} + b) — recurrent connection carries information across timesteps

2. **Vanilla RNN forward pass:**
   ```
   hₜ = tanh(Wₓₕ xₜ + Wₕₕ hₜ₋₁ + bₕ)
   yₜ = Wₕᵧ hₜ + bᵧ
   ```
   Variables: xₜ = input, hₜ = hidden state, yₜ = output, W = weight matrices, b = biases

3. **Vanishing gradient problem:**
   - During BPTT, gradients multiplied by Whh at each timestep
   - If |Whh| < 1: gradients → 0 exponentially (vanish) → early timesteps not updated
   - If |Whh| > 1: gradients → ∞ (explode)
   - Effect: RNNs fail to learn long-range dependencies (e.g., linking subject to verb 100 tokens apart)

4. **RNN vs LSTM comparison:**
   | | RNN | LSTM |
   |--|-----|------|
   | Memory | Hidden state only | Hidden state + cell state |
   | Long-term deps | Poor | Good (additive cell update) |
   | Complexity | Low | Higher (4× gates) |
   | Training stability | Poor (vanishing grads) | Better |

5. **LSTM components:**
   - **Forget gate** fₜ = σ(Wf[hₜ₋₁,xₜ]+bf): what to erase from cell state
   - **Input gate** iₜ = σ(Wi[hₜ₋₁,xₜ]+bi): what new info to write
   - **Cell candidate** g̃ₜ = tanh(Wg[hₜ₋₁,xₜ]+bg): candidate values to add
   - **Cell state** Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙g̃ₜ: gated update (additive → stable gradients)
   - **Output gate** oₜ = σ(Wo[hₜ₋₁,xₜ]+bo): controls what hidden state outputs
   - hₜ = oₜ ⊙ tanh(Cₜ)

## Coding Tasks

6. **RNN from scratch with NumPy:**
   ```python
   def rnn_forward(X, Wxh, Whh, Why, bh, by):
       h = np.zeros((hidden_size,))
       outputs = []
       for x in X:
           h = np.tanh(Wxh @ x + Whh @ h + bh)
           y = Why @ h + by
           outputs.append(y)
       return outputs, h
   ```

7. **Character-level RNN (PyTorch/TensorFlow):**
   - Encode chars as one-hot or embeddings
   - Many-to-many: predict next character at each step
   - Loss: cross-entropy over vocabulary

8. **LSTM for time series prediction:**
   - Input: windowed sequences of past values
   - Output: next value(s)
   - Architecture: `LSTM(hidden_size) → Linear(1)`
   - Loss: MSE

9. **LSTM for binary sequence classification (sentiment analysis):**
   - many-to-one: process full sequence, use last hidden state for classification
   - Architecture: `Embedding → LSTM → Linear(2) → Softmax`
   - Loss: binary cross-entropy

10. **Sequence modeling system design:**
    - Choose RNN or LSTM based on sequence length and dependency requirements
    - Consider bidirectional for classification (can see full context)
    - Consider stacked layers for complex patterns
    - Regularization: dropout on non-recurrent connections
