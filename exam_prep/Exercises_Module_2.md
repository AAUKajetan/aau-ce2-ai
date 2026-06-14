# Exercises — Module 2: MLP and CNN

## I. Theory

### 1. Backpropagation Derivation
- Chain rule applied backward through layers
- For output layer: δᴸ = ∇_a L ⊙ σ'(zᴸ)
- For hidden layers: δˡ = (Wˡ⁺¹ᵀ δˡ⁺¹) ⊙ σ'(zˡ)
- Weight gradients: ∂L/∂Wˡ = δˡ (aˡ⁻¹)ᵀ

### 2. Sigmoid Derivative
```
σ(x) = 1/(1+e^{-x})
σ'(x) = σ(x)(1 - σ(x))
```
Key insight: output range (0,1) so max gradient = 0.25 at x=0 → contributes to vanishing gradients

## II. Coding: MNIST with MLP

**Setup:** Use Chapter 12 of Python ML book. Dataset: MNIST.

1. **20 vs 40 hidden neurons** — same 1000 training images, cross-entropy, η=0.5, batch=10, 300 epochs
   - Expect: 40 hidden neurons → higher accuracy due to more capacity
2. **1000 vs 5000 training images** — 40 hidden, same hyperparams
   - Expect: more data → better generalization, lower test error
3. **Regularization λ ∈ {0.1, 1, 10}** — 40 hidden, 1000 images
   - Expect: moderate λ helps; too large → underfitting
4. **Heuristic hyperparameter search** — 40 hidden, 3000 images, find best η and batch size
   - Log accuracy per epoch, plot learning curves

## III. CNN Concepts

**Q: What is a CNN?**
- Convolutional layers: learn spatial feature detectors via shared-weight filters
- Pooling layers: downsample feature maps, add translation invariance
- Fully connected layers: final classification

**Output size formula** (no padding, stride=1):
```
output = (input - kernel + 1) / stride
For 32×32, kernel 3×3, stride 1: output = (32-3+1)/1 = 30 → 30×30
```

## IV. CNN Design Questions

- **Why pooling?** Reduces spatial dimensions, controls overfitting, adds invariance
- **Stride increases?** Larger stride → smaller output, fewer computations, coarser features
- **Fewer params than FC?** Weight sharing: one filter applied everywhere vs unique weight per connection
- **padding="valid"** → no padding, output smaller than input; **padding="same"** → zero-pad so output = input size

## V. Build CNN on MNIST

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

## VI. Manual Convolution (without Conv2D)

- Extract patches from input using nested loops or `np.lib.stride_tricks`
- Multiply element-wise with kernel, sum → one output pixel
- Loop over all positions (with stride and padding logic)
- Key: same result as Conv2D but educational about what convolution actually computes
