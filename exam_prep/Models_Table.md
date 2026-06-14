# All Models & Methods — ML and DNN Classes

## Classical ML / Feature Extraction

| Model | Full Name | Type | Key Idea | Module | Exercise |
|-------|-----------|------|---------|--------|---------|
| PCA | Principal Component Analysis | Unsupervised dim-reduction | Finds orthogonal directions of max variance; uses covariance matrix eigendecomposition | Module 1 | Exercise Module 1 |
| LDA | Linear Discriminant Analysis | Supervised dim-reduction | Maximizes between-class / within-class scatter ratio (Sw⁻¹SB); needs class labels | Module 1 | Exercise Module 1 |

---

## Neural Network Models

| Model | Full Name | Type | Key Idea | Module | Exercise |
|-------|-----------|------|---------|--------|---------|
| Perceptron | Perceptron | Linear classifier | sign(w·x + b); linearly separable only | Module 2 | — |
| Adaline | Adaptive Linear Neuron | Linear | MSE + linear activation; precursor to logistic regression | Module 2 | — |
| Logistic Regression | Logistic Regression | Linear classifier | Sigmoid output; still linear boundary | Module 2 | — |
| MLP | Multi-Layer Perceptron | Feedforward NN | Hidden layers + nonlinear activations; universal approximator | Module 2 | Exercise Module 2 |
| CNN | Convolutional Neural Network | Feedforward NN | Convolution + pooling; spatial feature detection; parameter sharing | Module 2 | Exercise Module 2 |
| RNN | Recurrent Neural Network | Recurrent NN | Hidden state hₜ carries memory across time steps; suffers vanishing gradients | Module 3 | Exercise Module 3 |
| LSTM | Long Short-Term Memory | Recurrent NN | Gates + cell state; additive update solves vanishing gradient | Module 3 | Exercise Module 3 |
| GRU | Gated Recurrent Unit | Recurrent NN | Simplified LSTM with 2 gates (reset, update); fewer parameters | Module 3 | — |
| AE | Autoencoder | Generative | Encoder → bottleneck z → Decoder; learns compressed representation | Module 9 | Exercise Module 4 |
| Denoising AE | Denoising Autoencoder | Generative | Corrupted input → reconstructs clean x; learns robust features | Module 9 | Exercise Module 4 |
| Sparse AE | Sparse Autoencoder | Generative | L1 penalty on z; encourages few active neurons | Module 9 | — |
| Contractive AE | Contractive Autoencoder | Generative | Jacobian penalty on encoder; smooth, stable latent space | Module 9 | — |
| VAE | Variational Autoencoder | Generative | Probabilistic latent z ~ N(μ,σ²); ELBO loss; reparameterization trick | Module 9 | Exercise Module 4 |
| β-VAE | Beta Variational Autoencoder | Generative | VAE with β·KL term; trades reconstruction quality for disentanglement | Module 9 | — |
| GAN | Generative Adversarial Network | Generative | Generator vs Discriminator; minimax game; Nash equilibrium | Module 10 | Exercise Module 5 |
| Transformer | Transformer | Attention-based | Self-attention + positional encoding + residuals; fully parallelizable | Module 10 | Exercise Module 5 |

---

## Reinforcement Learning Models

| Model | Full Name | Type | Key Idea | Module | Exercise |
|-------|-----------|------|---------|--------|---------|
| SARSA | State–Action–Reward–State–Action | Tabular, on-policy | TD control; uses actual next action aₜ₊₁ from ε-greedy policy | RL Module 1 | Exercise RL Module 2 |
| Q-learning | Q-learning | Tabular, off-policy | TD control; uses max over next actions; optimizes greedy policy | RL Module 1 | Exercise RL Module 2 |
| DQN | Deep Q-Network | Value-based, off-policy | Q-learning + NN + replay buffer + target network | RL Module 2 | Exercise RL Module 2 |
| DDQN | Double Deep Q-Network | Value-based, off-policy | DQN with decoupled action selection and evaluation; fixes overestimation bias | DRL Module 3 | Exercise DRL Module 3 |
| REINFORCE | REINFORCE | Policy gradient, on-policy | MC returns weighted by log-prob; unbiased but high variance | DRL Module 3 | Exercise DRL Module 3 |
| Actor-Critic | Actor-Critic | Policy gradient, on-policy | Actor πθ + Critic Vφ; TD error δₜ replaces MC return; lower variance | DRL Module 3 | Exercise DRL Module 3 |
| DDPG | Deep Deterministic Policy Gradient | Actor-critic, off-policy | Deterministic actor μθ; continuous actions; soft target updates (Polyak) | RL Module 4 | Exercise RL Module 4 |
| PPO | Proximal Policy Optimization | Actor-critic, on-policy | Stochastic actor; clipped probability ratio; GAE advantages | RL Module 4 | Exercise RL Module 4 |
| VDN | Value Decomposition Networks | MARL, cooperative | Q_tot = Σ Qᵢ; additive Q factorization; CTDE | RL Module 5 | — |
| QMIX | Q Mixing Network | MARL, cooperative | Monotonic mixing network conditioned on global state; IGM condition | RL Module 5 | — |
| COMA | Counterfactual Multi-Agent Policy Gradients | MARL, cooperative | Centralized critic; counterfactual baseline isolates each agent's contribution | RL Module 5 | — |
| MADDPG | Multi-Agent Deep Deterministic Policy Gradient | MARL, mixed | DDPG with centralized critic per agent; decentralized actor | RL Module 5 | — |
| MAPPO | Multi-Agent Proximal Policy Optimization | MARL, cooperative | PPO with shared centralized critic on global state; strong cooperative baseline | RL Module 5 | — |
