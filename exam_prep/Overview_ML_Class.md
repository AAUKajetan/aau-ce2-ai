# ML Class — Exam Overview

> Source modules: Module 1 + Exercises Module 1

---

## Topics Map

| Topic | Key File |
|-------|---------|
| AI/ML/DL taxonomy | Module_1.md |
| Types of ML | Module_1.md |
| Data preprocessing pipeline | Module_1.md |
| PCA (unsupervised dim-reduction) | Module_1.md + Exercises_Module_1.md |
| LDA (supervised dim-reduction) | Module_1.md + Exercises_Module_1.md |
| Train/val/test splits & cross-validation | Module_1.md |

---

## 1. AI / ML / DL Taxonomy

- **AI** ⊃ **ML** ⊃ **Deep Learning**
- ML: systems that learn from data instead of explicit rules
- DL: subset of ML using multi-layer neural networks

**Types of ML:**

| Type | Labels? | Goal | Examples |
|------|---------|------|---------|
| Supervised | Yes | Learn f(X)→Y | Classification, regression |
| Unsupervised | No | Find structure in X | Clustering, dim-reduction |
| Semi-supervised | Partial | Combine both | Self-training, label propagation |
| Reinforcement | Rewards | Agent maximizes cumulative reward | Game playing, robotics |

---

## 2. Data Preprocessing Pipeline

**Order: Clean → Transform → Select/Extract → Split**

### Cleaning
- Handle missing values (imputation or removal)
- Remove duplicates, fix inconsistencies
- Detect and treat outliers

### Transformation
- **Normalization** (min-max): scale to [0, 1]
- **Standardization** (z-score): zero mean, unit variance — preferred for approximately Gaussian data
- **Encoding**: one-hot for nominal categories; ordinal encoding for ordered categories

### Feature Selection vs Feature Extraction
- **Selection**: pick a subset of original features (filter / wrapper / embedded methods)
- **Extraction**: create new features from originals (PCA, LDA)

### Partitioning
- Train / Validation / Test (e.g., 60/20/20)
- **k-Fold Cross-Validation**: rotate k folds → more robust evaluation; stratified splits preserve class balance

### Common Pitfalls
- **Data leakage**: preprocessing on full dataset before splitting → invalid evaluation
- **Class imbalance**: oversample minority / undersample majority / use class weights
- Feature scaling required for distance-based and gradient-based models

---

## 3. PCA (Principal Component Analysis)

**Unsupervised, linear dimensionality reduction.**

Goal: find orthogonal directions of maximum variance (principal components).

### Algorithm
1. Center data: `X ← X − mean(X)`
2. Compute covariance matrix: `C = XᵀX / (n-1)`
3. Eigendecomposition: `C = Q Λ Qᵀ`
4. Sort eigenvectors by descending eigenvalue
5. Select top-k eigenvectors → transformation matrix W (shape d×k)
6. Project: `Z = X @ W`

### Key Facts
- No class labels used — purely structural
- Eigenvectors = principal components; eigenvalues = explained variance
- Plot cumulative variance ratio to choose k: `cumsum(sorted eigenvalues) / sum(eigenvalues)`
- Max components = min(n-1, d)

---

## 4. LDA (Linear Discriminant Analysis)

**Supervised, linear dimensionality reduction.**

Goal: maximize between-class scatter / within-class scatter ratio.

### Matrices
- **Sw** (within-class scatter): `Σ_c Σ_{x∈c} (x − μ_c)(x − μ_c)ᵀ`
- **SB** (between-class scatter): `Σ_c n_c (μ_c − μ)(μ_c − μ)ᵀ`
- Solve generalized eigenvalue problem on **Sw⁻¹ SB**

### Key Facts
- Uses class labels → maximizes class separability
- Max discriminant components = **C − 1** (for C classes)
- On Iris (3 classes) → at most 2 LDA components

---

## 5. PCA vs LDA — Quick Comparison

| | PCA | LDA |
|--|-----|-----|
| Supervised? | No | Yes |
| Goal | Max variance | Max class separation |
| Matrix used | Covariance | Sw⁻¹ SB |
| Max components | min(n-1, d) | C − 1 |

---

## 6. Exam Question Patterns (from exercises)

- Define AI / ML / DL — explain hierarchy
- Supervised vs unsupervised: give examples of each
- What problems does ML solve better than hand-coded rules?
- Difference between algorithm and model
- Steps of PCA from scratch (covariance → eigen → sort → project)
- Steps of LDA — how Sw and SB are computed
- Why must you standardize before PCA?
- Data leakage — what is it, why does it matter?
- k-Fold cross-validation — why use it vs a single split?

---

## 7. Abbreviations

| Abbr | Meaning | Topic |
|------|---------|-------|
| AI | Artificial Intelligence | §1 |
| ML | Machine Learning | §1 |
| DL | Deep Learning | §1 |
| RL | Reinforcement Learning | §1 |
| PCA | Principal Component Analysis | §3 |
| LDA | Linear Discriminant Analysis | §4 |
| Sw | Within-class scatter matrix | §4 |
| SB | Between-class scatter matrix | §4 |
