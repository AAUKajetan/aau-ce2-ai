# Exercises — Module 1: AI/ML Foundations

## Task 1: Theory Questions

1. What is the difference between AI, ML, and deep learning?
   - AI: broad field of intelligent machines; ML: subset where systems learn from data; DL: subset of ML using multi-layer neural networks
2. What is machine learning?
   - Systems that learn patterns from data without being explicitly programmed
3. Unsupervised vs supervised learning?
   - Supervised: labeled data, learns input→output mapping; Unsupervised: no labels, finds structure
4. What is reinforcement learning?
   - Agent learns via trial-and-error interactions with an environment using reward signals
5. What problems does ML solve better than traditional programming?
   - Pattern recognition in high-dim data, tasks where rules are hard to hand-code (image/speech/NLP)
6. Unsupervised problems: clustering, dimensionality reduction, anomaly detection, density estimation
7. Supervised problems: classification, regression, object detection, translation
8. Algorithm vs model: algorithm is the training procedure; model is the learned artifact (parameters)
9. ML challenges: insufficient/noisy data, overfitting, underfitting, feature engineering, distribution shift, interpretability

## Task 2: Python Environment Setup

- Install: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tensorflow`/`torch`

## Task 3: PCA on Iris Dataset

**Steps:**
1. Load Iris (all 4 features with Pandas), standardize (zero mean, unit variance), split 70/30
2. Compute covariance matrix manually: `C = (X.T @ X) / (n-1)`; compute eigenvalues/eigenvectors via `np.linalg.eig(C)`
3. Plot cumulative variance ratio: `np.cumsum(sorted_eigenvalues) / sum(eigenvalues)`
4. Select top k=3 eigenvectors → transformation matrix W (shape d×k)
5. Project: `Z = X @ W`, plot 2D/3D scatter of projected data

**Key concepts:**
- PCA is unsupervised — uses covariance matrix, not class labels
- Eigenvectors = principal components; eigenvalues = explained variance
- Sort eigenvectors by descending eigenvalue before selecting top k

## Task 4: LDA on Iris

**Steps:** Same pipeline but use Sw⁻¹SB instead of the covariance matrix

- **Sw** (within-class scatter): sum over classes of scatter matrix of class samples
- **SB** (between-class scatter): weighted sum of (μc - μ)(μc - μ)ᵀ
- Max components = C-1 = 2 for Iris (3 classes)
- LDA is supervised — uses class labels to maximize class separability

**PCA vs LDA:**
| | PCA | LDA |
|--|-----|-----|
| Supervised | No | Yes |
| Criterion | Max variance | Max class separation |
| Max components | min(n-1, d) | C-1 |
| Matrix | Covariance | Sw⁻¹SB |
