# Exercises — DRL Module 3: DQN, DDQN, REINFORCE, Actor-Critic

**Environment:** CartPole-v1 for all exercises (optional: Acrobot-v1)
**Stack:** Python 3.10+, gymnasium[classic-control], torch, numpy, matplotlib
**Minimum:** 3 random seeds per experiment; report learning curves + interpretation

---

## Exercise 1: Rebuild and Analyze DQN

**Goal:** Connect DQN equations to code; verify role of replay, target nets, TD target.

**Implement:**
- Online QNet Qθ(s,a) and target QNet Qθ⁻(s,a)
- Replay buffer storing (s,a,r,s',done)
- ε-greedy action selection
- Mini-batch training with: `yₜ = rₜ₊₁ + γ(1-dₜ) max_a' Qθ⁻(sₜ₊₁, a')`
- Hard target copy every C steps
- Log episode returns and training loss

**Comparison variant:** Disable replay OR disable target net and compare learning behavior.

**Report questions:**
1. Why does replay help SGD? (Breaks temporal correlation between consecutive samples)
2. Why does target net reduce moving-target problem? (Fixes the Q target for C steps so gradient is stable)
3. What changed in learning curves when removed? (Expect instability/oscillation/divergence)

**Common bugs:** Forgetting terminal masking, updating target every gradient step, computing target with online net.

---

## Exercise 2: Convert DQN → DDQN

**Goal:** One-line change reveals how target design affects overestimation bias.

**DDQN target:**
```
yₜ = rₜ₊₁ + γ(1-dₜ) · Qθ⁻(sₜ₊₁, argmax_a' Qθ(sₜ₊₁, a'))
```
- Online net θ selects action; target net θ⁻ evaluates it
- Decouples selection from evaluation → reduces overestimation bias

**Steps:**
1. Reuse all DQN components (buffer, architecture, optimizer, target sync)
2. Change only the target computation (2 lines of code)
3. Track average predicted Q-value during training as a diagnostic
4. Compare DQN vs DDQN across 3+ seeds

**Report questions:**
1. DQN selection+evaluation uses same net → systematically picks overestimated action
2. DDQN: one line changes `target_qnet(s2).max()` to `target_qnet(s2).gather(1, qnet(s2).argmax(...))`
3. DDQN should show less optimistic Q-values, potentially more stable learning
4. CartPole may solve too quickly to show much difference — try Acrobot or reduce network size

---

## Exercise 3: REINFORCE and Variance Study

**Goal:** Stochastic policy, Monte Carlo returns, understand high variance.

**Algorithm:**
1. Collect full episode trajectory under πθ
2. Compute returns backward: Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ...
3. Actor loss: `L = -Σₜ log πθ(aₜ|sₜ) · Gₜ`
4. Single gradient update after full episode

**Policy network:** Linear(4→128)→ReLU→Linear(128→2)→Softmax
- Sample action from Categorical distribution; store `log_prob = dist.log_prob(action)`

**Variance reduction (implement one):**
- **Return normalization:** `G_norm = (G - G.mean()) / (G.std() + 1e-8)` within each episode
- **Baseline:** subtract running average of returns: `G - baseline`; use exponential moving average

**Compare:** Original REINFORCE vs variance-reduced version (same seeds)

**Report questions:**
1. On-policy: uses data only from current policy πθ; trajectories must be fresh
2. High variance: full MC return Gₜ includes noise from all future rewards; single episode is noisy
3. Normalization/baseline should make curve smoother and learn faster
4. `log πθ(aₜ|sₜ)` in code = `dist.log_prob(action)` at the sampled action index

**Common bugs:** Storing only action index but not log_prob; forgetting that reward at t+1 contributes to Gₜ.

---

## Exercise 4: One-Step Actor-Critic

**Goal:** Replace MC return with TD error for lower-variance updates.

**Components:**
- Actor: πθ(a|s) → action logits
- Critic: Vφ(s) → scalar value estimate
- TD target: `yₜ = rₜ₊₁ + γ(1-dₜ)Vφ(sₜ₊₁)`
- TD error: `δₜ = yₜ - Vφ(sₜ)` (detached from actor gradient)
- Critic loss: `(Vφ(sₜ) - yₜ)²`
- Actor loss: `-log πθ(aₜ|sₜ) · δₜ`

**Key difference from REINFORCE:** Updates every step, not after full episode. Critic provides a low-variance baseline.

**Report questions:**
1. Critic provides δₜ = how much better/worse the action was than expected → reduces variance
2. TD error variance < MC return variance (single step vs full trajectory noise)
3. Actor-critic usually more stable, faster convergence than REINFORCE
4. Positive δₜ: action led to better-than-expected outcome → increase probability; negative: decrease

---

## Optional: Comparison Table

| Algorithm | On/Off policy | Replay | Target net | Learning signal | Variance |
|-----------|--------------|--------|------------|-----------------|----------|
| DQN | Off | Yes | Yes | TD target | Low |
| DDQN | Off | Yes | Yes | TD target (decoupled) | Low |
| REINFORCE | On | No | No | MC return Gₜ | High |
| Actor-Critic | On | No | No | TD error δₜ | Medium |
