# Exercises — RL Module 4: DDPG and PPO (Continuous Control)

**Environment:** Pendulum-v1 (continuous torque action, 3D state: [cos θ, sin θ, θ̇])
**Stack:** gymnasium[classic-control], torch, numpy, matplotlib
**Install:** `pip install gymnasium[classic-control] torch numpy matplotlib`

---

## Part 0: Inspect the Continuous-Control Environment

```python
env = gym.make("Pendulum-v1")
state, _ = env.reset(seed=42)
# State dim: 3, Action dim: 1, Action in [-2, 2]
```

**Key difference from CartPole:** Action is a real-valued torque — you cannot `argmax Q(s,a)` over a continuous space. This is why DQN doesn't apply here.

**Transition tuple:** (sₜ, aₜ, rₜ₊₁, sₜ₊₁, doneₜ) where aₜ is a float vector, not integer.

---

## Part I: DDPG Implementation

**Core idea:**
```
aₜ = μθ(sₜ) + εₜ          (data collection with noise)
yₜ = rₜ₊₁ + γ(1-done)·Qφ⁻(sₜ₊₁, μθ⁻(sₜ₊₁))   (critic target)
```

### Networks

```python
# Actor: state → bounded action via tanh * action_limit
DDPGActor: Linear(3,128)→ReLU→Linear(128,128)→ReLU→Linear(128,1)→Tanh → scale by 2.0

# Critic: (state, action) concatenated → scalar Q value
DDPGCritic: Linear(3+1,128)→ReLU→Linear(128,128)→ReLU→Linear(128,1)
```

### Training Steps

1. **Collect action** with Gaussian noise: `a = clip(μθ(s) + noise_std * randn(), low, high)`
2. **Store** transition in replay buffer
3. **Compute critic target:** `y = r + γ(1-done)·Qφ⁻(s', μθ⁻(s'))`  (no_grad)
4. **Update critic:** minimize `MSE(Qφ(s,a), y)`
5. **Update actor:** minimize `-Qφ(s, μθ(s)).mean()`  (gradient ascent on Q)
6. **Soft update targets:** `θ⁻ ← τθ + (1-τ)θ⁻`  (τ=0.005)

### TODO Map (starter file)
| TODO | Location | What to understand |
|------|----------|--------------------|
| 1-2 | ddpg_update() | Target actor + critic produce bootstrapped DDPG target |
| 3 | ddpg_update() | Online critic predicts Q(s,a) for sampled transitions |
| 4 | ddpg_update() | Actor loss = -critic score on actor's own actions |
| 5 | ddpg_update() | Soft update: slow target tracking vs hard copy |

### Common Bugs
- Adding noise during evaluation (should be noise-free)
- Not clipping actions to environment bounds
- Bootstrapping through terminal states (terminal: y = r, no γ·Q term)
- Shape mismatch: (batch, action_dim) vs (batch,)

### What to Plot
- Evaluation return vs environment steps
- Critic loss vs update step
- Actor loss vs update step

### Ablation Experiments
- **Exploration noise scale:** compare noise_std ∈ {0.05, 0.15, 0.5}
- **Target update rate τ:** compare τ ∈ {0.001, 0.005, 0.05}

---

## Part II: PPO Implementation

**Core idea:** Stochastic actor, state-value critic V(s), fresh rollouts, clipped update.

```
L^clip(θ) = E[min(ρₜ(θ)·Âₜ, clip(ρₜ(θ), 1-ε, 1+ε)·Âₜ)]
ρₜ(θ) = πθ(aₜ|sₜ) / πθ_old(aₜ|sₜ)
```

### Networks

```python
# Gaussian actor: state → (mean, std) for Gaussian action distribution
GaussianActor: Linear(3,128)→Tanh→Linear(128,128)→Tanh→Linear(128,1)
               + learnable log_std parameter

# Value critic: state → scalar V(s)
ValueCritic: Linear(3,128)→Tanh→Linear(128,128)→Tanh→Linear(128,1)
```

### Training Steps

1. **Collect rollout:** store (s, a, r, done, old_log_prob, V(s)) for N steps
2. **Compute GAE advantages** backward through time:
   ```
   δₜ = rₜ + γ(1-doneₜ)·V(sₜ₊₁) - V(sₜ)
   Âₜ = δₜ + γλ(1-doneₜ)·Âₜ₊₁    (backward pass, last_gae=0)
   returns = advantages + values
   ```
3. **Normalize advantages:** `Â = (Â - mean) / (std + 1e-8)`
4. **Multiple PPO epochs** on same rollout batch:
   - Recompute `new_log_probs` from current actor
   - Compute ratio: `ρ = exp(new_log_probs - old_log_probs)`
   - Clipped actor loss: `-min(ρ·Â, clip(ρ, 1-ε, 1+ε)·Â).mean()`
   - Value loss: `MSE(V(s), returns)`
5. **Discard rollout**, collect fresh data

### TODO Map (starter file)
| TODO | Location | What to understand |
|------|----------|--------------------|
| 6-7 | compute_gae_and_returns() | TD residuals → GAE advantages and returns |
| 8-9 | ppo_update() | Probability ratio computation and clipping |
| 10 | ppo_update() | Value critic regression loss |

### Common Bugs
- Forgetting to store `old_log_probs` during rollout
- Not recomputing `new_log_probs` during optimization (must use current actor)
- Using returns where advantages should be in the actor loss
- Mixing rollout data from different policies

### Ablation Experiments
- **Clipping:** compare ε ∈ {0.1, 0.2, 0.3}
- **GAE λ:** compare λ ∈ {0.95, 1.0} (λ=1.0 is MC return, high variance)

---

## Part III: DDPG vs PPO Comparison

| Aspect | DDPG | PPO |
|--------|------|-----|
| Actor | Deterministic μθ(s) | Stochastic πθ(a\|s) ~ N(μ,σ) |
| Critic | Q(s,a) | V(s) |
| Data | Replay buffer (off-policy) | Fresh rollouts (on-policy) |
| Exploration | External noise εₜ | Stochasticity + entropy |
| Target nets | Yes (actor + critic) | No |
| Update | Per step | Multiple epochs per rollout |

**Conceptual questions:**
- Why replay buffer natural for DDPG but not PPO? DDPG is off-policy → old data valid; PPO is on-policy → must use data from current policy
- Why does DDPG need target actor AND critic? Both used in DDPG target: y = r + γQ_{φ⁻}(s', μ_{θ⁻}(s')); without target actor, coupling between actor and critic updates
- Why does PPO compare new vs old policy? To bound how much the policy changes per update — large changes can destroy performance (trust region principle)
- Positive advantage in PPO: this (s,a) pair led to better-than-expected return → increase probability
