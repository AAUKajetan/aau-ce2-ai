# RL Class — Full Exam Study Overview

> Covers all 5 RL modules. Key formulas, comparisons, exercise insights, exam question patterns.

---

## Module 1: Foundations — MDP, Bellman, DP, MC, TD, SARSA, Q-learning

### The Agent–Environment Loop
At each step t: agent observes sₜ → selects aₜ ~ π(·|sₜ) → environment returns rₜ and sₜ₊₁ ~ P(·|sₜ,aₜ).

### Markov Decision Process (MDP)
Formal tuple **(S, A, P, R, γ)**:
- S: state space; A: action space
- P(s'|s,a): transition probability
- R(s,a): expected immediate reward
- γ ∈ [0,1): discount factor

**Markov Property:** sₜ₊₁ ⊥ history | sₜ — current state is sufficient.

### Returns and Discounting
```
Gₜ = rₜ + γ rₜ₊₁ + γ² rₜ₊₂ + ... = Σₖ γᵏ rₜ₊ₖ
```
- γ→0: myopic (immediate reward only)
- γ→1: far-sighted (future matters equally)
- Discounting guarantees finite returns in continuing tasks

### Value Functions
```
Vπ(s) = Eπ[Gₜ | sₜ=s]
Qπ(s,a) = Eπ[Gₜ | sₜ=s, aₜ=a]
Vπ(s) = Σₐ π(a|s) Qπ(s,a)
```

### Bellman Equations

**Bellman Expectation (evaluates policy π):**
```
Vπ(s) = Σₐ π(a|s) Σₛ' P(s'|s,a) [R(s,a) + γ Vπ(s')]
Qπ(s,a) = Σₛ' P(s'|s,a) [R(s,a) + γ Σₐ' π(a'|s') Qπ(s',a')]
```

**Bellman Optimality (finds optimal policy):**
```
V*(s) = max_a Σₛ' P(s'|s,a) [R(s,a) + γ V*(s')]
Q*(s,a) = Σₛ' P(s'|s,a) [R(s,a) + γ max_a' Q*(s',a')]
π*(s) = argmax_a Q*(s,a)
```

**Exam question:** "What is the difference between Bellman expectation and optimality?"
- Expectation: *evaluates* a given policy (asks "how good is π?")
- Optimality: *finds* the best policy (asks "what is the best possible?")

### Dynamic Programming (requires full model)
| Method | Description |
|--------|-------------|
| Policy Evaluation | Iteratively apply Bellman expectation until Vπ converges |
| Policy Improvement | π'(s) = argmax_a Q(s,a) — greedy update |
| Policy Iteration | Alternate evaluation and improvement until stable |
| Value Iteration | Apply Bellman optimality directly: V(s) ← max_a [R(s,a) + γΣP·V(s')] |

### Monte Carlo (MC) Methods
- Sample **full episodes**, compute returns Gₜ from real trajectories
- Update: `V(sₜ) ← V(sₜ) + α(Gₜ − V(sₜ))`
- **No model needed**, **unbiased**, but **high variance** (entire episode is noisy)
- Requires episodic tasks (must reach terminal state)

### Temporal Difference (TD) Learning

**TD(0) — one-step bootstrap:**
```
V(sₜ) ← V(sₜ) + α(rₜ + γV(sₜ₊₁) − V(sₜ))
```
TD target: `rₜ + γV(sₜ₊₁)` — lower variance than MC, but **biased** (bootstraps from current estimate)

**n-Step Returns:**
```
Gₜ⁽ⁿ⁾ = rₜ + γrₜ₊₁ + ... + γⁿ⁻¹rₜ₊ₙ₋₁ + γⁿV(sₜ₊ₙ)
```
n=1: TD(0), n=∞: MC — controls bias/variance tradeoff

### Comparison: DP vs MC vs TD
| | DP | MC | TD |
|--|----|----|-----|
| Model required | Yes | No | No |
| Full episode | N/A | Yes | No |
| Bias | None | None | Yes (bootstrap) |
| Variance | Low | High | Low |
| Updates | Exact sweep | End of episode | Each step |

### Exploration
- **ε-greedy:** with prob ε → random action; else argmax Q
- **GLIE** (Greedy in the Limit with Infinite Exploration): ε→0 as t→∞ — needed for convergence guarantees

### SARSA vs Q-learning

**SARSA (on-policy TD control):**
```
Q(sₜ,aₜ) ← Q(sₜ,aₜ) + α[rₜ + γQ(sₜ₊₁,aₜ₊₁) − Q(sₜ,aₜ)]
```
Uses the *actual next action* aₜ₊₁ sampled from the behavior policy.

**Q-learning (off-policy TD control):**
```
Q(sₜ,aₜ) ← Q(sₜ,aₜ) + α[rₜ + γ max_a Q(sₜ₊₁,a) − Q(sₜ,aₜ)]
```
Uses *max* over next actions — optimizes the greedy (target) policy regardless of behavior.

**Key exam distinction (CliffWalking):**
- SARSA learns a *safer* path — accounts for ε-random exploration risk
- Q-learning learns the *optimal* (cliff-edge) path — ignores exploration policy
- Q-learning falls off cliff more during training; SARSA is safer but suboptimal under exploration

### Function Approximation
When state space is large/continuous: use parameterized Qθ(s,a) (neural net).
**Deadly triad**: function approximation + bootstrapping + off-policy → potential instability.

---

## Module 2: Deep Q-Networks (DQN)

### Core Idea
Tabular Q-learning fails for large/continuous state spaces. DQN = Q-learning + neural network + **two stabilizing tricks**.

### Problem 1: Correlated Samples → **Experience Replay**
- Store transitions (s,a,r,s',done) in circular buffer of size N (e.g., 1M)
- Sample **random mini-batches** for each update
- Breaks temporal correlations; enables data reuse (sample efficiency)

### Problem 2: Moving Target → **Target Network θ⁻**
- Separate copy of Q-network with **frozen** parameters
- Updated by hard copy θ→θ⁻ every C steps (e.g., 1000–10000)
- Stabilizes the TD target during learning

### DQN Loss
```
L(θ) = E[(yₜ − Qθ(sₜ,aₜ))²]
yₜ = rₜ + γ max_a' Qθ⁻(sₜ₊₁, a')    (0 if terminal)
```
Gradient flows only through online network θ; target yₜ is treated as constant.

### Terminal Masking (critical!)
```
yₜ = rₜ + (1 − doneₜ) · γ · max_a' Qθ⁻(sₜ₊₁, a')
```
If done=True: no bootstrapping — the episode is over.

### Loss Choices
| Loss | When to prefer |
|------|----------------|
| MSE: (y−Q)² | Sensitive to large errors |
| Huber (smooth L1) | Robust to outliers — preferred in RL |

### DQN Training Loop
```
for each step:
    aₜ = ε-greedy(Qθ(sₜ))
    execute aₜ → store (sₜ, aₜ, rₜ, sₜ₊₁, doneₜ)
    if buffer ≥ min_size:
        sample mini-batch
        compute yₜ using θ⁻
        loss ← Huber(yₜ − Qθ(sₜ,aₜ))
        backprop, update θ
    every C steps: θ⁻ ← θ
    decay ε
```

### Ablation Effects (exam question pattern)
| Remove | Effect |
|--------|--------|
| Replay buffer | Highly correlated updates → instability, slower convergence |
| Target network | Moving target → oscillation / divergence |
| Both | Likely divergence |
| Huber → MSE | More sensitive to large TD errors, less robust |
| Adam → SGD | Needs careful lr tuning, may converge slower |

### Hyperparameter Reference
| Hyperparameter | Typical Value |
|----------------|--------------|
| Replay buffer size | 10⁵ – 10⁶ |
| Mini-batch size | 32–64 |
| Target update freq | 1000–10000 steps |
| ε start/end/decay | 1.0 / 0.01 / 10⁵ steps |
| Learning rate | 1e-4 – 2.5e-4 |
| Discount γ | 0.99 |

---

## Module 3: DDQN and Policy Gradient Methods

### Overestimation Bias in DQN
`max_a' Qθ⁻(sₜ₊₁,a')` uses the **same network** to both *select* and *evaluate* the best action. Noise in Q-estimates means max always picks the overestimated action → systematic positive bias accumulates.

### Double DQN (DDQN) — Fix
**Decouple selection and evaluation:**
```
a* = argmax_a Qθ(sₜ₊₁, a)          ← online net selects
yₜ = rₜ + γ Qθ⁻(sₜ₊₁, a*)          ← target net evaluates
```
Drop-in replacement — same architecture, same hyperparameters, just change 2 lines.

**In code:**
```python
a_star = qnet(s2).argmax(dim=1, keepdim=True)
y = r + (1-done) * gamma * target_qnet(s2).gather(1, a_star)
```

### Limitations of Value-Based Methods (why we need PG)
1. **Discrete actions only** — argmax over Q doesn't work for continuous action spaces
2. **Deterministic policy** — cannot represent stochastic policies
3. **Indirect** — policy is implicit (extracted via argmax)

### Policy Gradient Theorem
```
∇θ J(θ) = Eπθ [ ∇θ log πθ(aₜ|sₜ) · Gₜ ]
```
- `∇θ log πθ(a|s)`: direction to increase probability of a in state s
- Weighted by Gₜ: reinforce good trajectories, suppress bad ones

### REINFORCE Algorithm
```
for each episode:
    collect trajectory (s₀,a₀,r₀,...,sT)
    for each t:
        Gₜ = Σₖ γᵏ rₜ₊ₖ          ← compute backward
        θ ← θ + α ∇θ log πθ(aₜ|sₜ) · Gₜ
```
- **Unbiased** estimate, but **high variance** (full episode returns are noisy)
- On-policy: must use fresh data from current policy

### Variance Reduction via Baseline
```
∇θ J(θ) = E [ ∇θ log πθ(a|s) · (Gₜ − b(sₜ)) ]
```
Baseline b(s) does **not** change the expected gradient (it cancels). Natural choice: b(sₜ) = Vπ(sₜ).

**Advantage function:**
```
Aₜ = Qπ(sₜ,aₜ) − Vπ(sₜ)
```
Measures how much better action aₜ is than the *average* action in state sₜ.
- Aₜ > 0 → increase π(a|s); Aₜ < 0 → decrease π(a|s)

### Actor-Critic Architecture
- **Actor** πθ(a|s): policy network — updated via policy gradient using advantage
- **Critic** Vφ(s): value network — provides baseline, trained via TD error
- TD target: `yₜ = rₜ + γ(1−doneₜ)Vφ(sₜ₊₁)`
- TD error (advantage): `δₜ = yₜ − Vφ(sₜ)`
- Actor loss: `-log πθ(aₜ|sₜ) · δₜ`; Critic loss: `(Vφ(sₜ) − yₜ)²`
- Enables **online (per-step) updates** rather than waiting for full episodes

### Algorithm Comparison (Module 3)
| Algorithm | On/Off | Replay | Target Net | Learning Signal | Variance |
|-----------|--------|--------|------------|-----------------|----------|
| DQN | Off | Yes | Yes | TD target | Low |
| DDQN | Off | Yes | Yes | TD target (decoupled) | Low |
| REINFORCE | On | No | No | MC return Gₜ | High |
| Actor-Critic | On | No | No | TD error δₜ | Medium |

---

## Module 4: DDPG and PPO (Continuous Control)

### Why Continuous Control is Different
For continuous actions aₜ ∈ ℝⁿ: argmax over Q-values is **intractable**. Need a different approach.

---

### DDPG (Deep Deterministic Policy Gradient)
**Key idea:** Deterministic actor μθ(s) ∈ ℝⁿ → can differentiate *through* it.

**Deterministic Policy Gradient theorem (Silver et al. 2014):**
```
∇θ J ≈ E [ ∇θ μθ(s) · ∇a Qφ(s,a)|_{a=μθ(s)} ]
```

**4 Networks:**
| Network | Params | Purpose |
|---------|--------|---------|
| Online actor | θ | Selects actions μθ(s) |
| Target actor | θ⁻ | Stable action for critic target |
| Online critic | φ | Estimates Q(s,a) |
| Target critic | φ⁻ | Computes TD target |

**Soft Target Updates (Polyak averaging):**
```
θ⁻ ← τθ + (1−τ)θ⁻     τ ≪ 1 (e.g., 0.005)
```
Smoother tracking than hard copy every C steps.

**Critic target:**
```
yₜ = rₜ + γ Qφ⁻(sₜ₊₁, μθ⁻(sₜ₊₁))    (0 if terminal)
```

**Actor update (gradient ascent on Q):**
```
∇θ J ≈ (1/N) Σ ∇a Qφ(s,a)|_{a=μθ(s)} · ∇θ μθ(s)
```

**Exploration:** Deterministic policy has no intrinsic exploration. Add noise during training:
```
aₜ = μθ(sₜ) + Nₜ
```
- **Ornstein-Uhlenbeck (OU) noise**: temporally correlated — smooth exploration for physical systems
- **Gaussian noise**: simpler, also effective

---

### PPO (Proximal Policy Optimization)
**Key idea:** Constrain how much the policy can change per update. Prevents destructive large updates.

**Probability ratio:**
```
rₜ(θ) = πθ(aₜ|sₜ) / πθ_old(aₜ|sₜ)
```

**Clipped objective:**
```
L^CLIP(θ) = E[ min( rₜ(θ)Âₜ,  clip(rₜ(θ), 1−ε, 1+ε)Âₜ ) ]
```
- Âₜ > 0: increase prob of aₜ, but not more than (1+ε)×old
- Âₜ < 0: decrease prob of aₜ, but not less than (1−ε)×old
- Typical ε = 0.2

**Full PPO loss:**
```
L(θ) = L^CLIP(θ) − c₁ L^VF(θ) + c₂ H[πθ]
```
- L^VF: value loss — (V(s) − Vₜarget)²
- H[πθ]: entropy bonus — encourages exploration, prevents policy collapse
- c₁ ≈ 0.5, c₂ ≈ 0.01

**GAE (Generalized Advantage Estimation):**
```
δₜ = rₜ + γV(sₜ₊₁) − V(sₜ)      ← TD residual
Âₜ = Σₗ (γλ)ˡ δₜ₊ₗ
```
- λ=0: pure TD(0) — low variance, high bias
- λ=1: full MC return — high variance, low bias
- Typical λ = 0.95

**PPO Training Loop:**
```
for each rollout:
    collect N transitions with πθ_old
    compute GAE advantages (backward)
    normalize advantages
    for K epochs:
        sample mini-batches
        compute clipped loss
        update θ
    θ_old ← θ
```

---

### DDPG vs PPO Comparison
| | DDPG | PPO |
|--|------|-----|
| Policy type | Deterministic | Stochastic Gaussian |
| On/Off policy | Off-policy | On-policy |
| Action space | Continuous only | Discrete or continuous |
| Exploration | Explicit noise (OU/Gaussian) | Entropy bonus + stochasticity |
| Sample efficiency | Higher | Lower |
| Stability | Moderate | High |
| Target networks | Yes (actor + critic) | No |
| Replay buffer | Yes | No (rollout buffer) |
| Hyperparameter sensitivity | High | Lower |

**Exam question:** "Why does DDPG need target actor AND critic?"
- Critic target uses: `y = r + γQφ⁻(s', μθ⁻(s'))` — both target nets prevent coupling between actor and critic updates.

**Exam question:** "Why can't you use a replay buffer in PPO?"
- PPO is on-policy: policy gradient estimate is only valid for data collected under the *current* policy.

---

## Module 5: Multi-Agent Reinforcement Learning (MARL)

### Stochastic Games (MDP Extension)
**(S, A₁,...,Aₙ, P, R₁,...,Rₙ, γ)**
- N agents with joint action **a** = (a₁,...,aₙ)
- Transition: s' ~ P(·|s,**a**)
- Agents may observe only partial state (partial observability → Dec-POMDP)

### Settings by Reward Structure
| Setting | Rewards | Goal | Example |
|---------|---------|------|---------|
| Cooperative | Shared team reward | Maximize joint return | Multi-robot coordination |
| Competitive | Zero-sum | Maximize own, minimize others | Chess, RTS |
| Mixed | Individual (partly aligned) | Individual returns | Autonomous driving |

### Core Challenges

**1. Non-Stationarity:** From agent i's perspective, other agents are part of the environment. As other agents learn, the effective transition dynamics change → breaks convergence guarantees of single-agent algorithms.

**2. Credit Assignment:** With shared team reward, which agent's action caused which outcome? Hard to attribute in large teams.

**3. Partial Observability:** Agents see only oᵢ, not full state s → Decentralized Partially Observable MDP (Dec-POMDP).

**4. Joint Action Space Explosion:** |A₁| × ... × |Aₙ| — exponential in N. Makes centralized Q(s,**a**) intractable.

### CTDE (Centralized Training, Decentralized Execution)
The dominant paradigm in cooperative MARL:
- **Training:** access to global state, all observations/actions
- **Execution:** each agent uses only local observation oᵢ

Allows rich credit assignment during training; scalable at test time.

### Value Factorization Methods

**VDN (Value Decomposition Networks):**
```
Q_tot(s,a) = Σᵢ Qᵢ(oᵢ,aᵢ)
```
Global Q = sum of individual Qs. Strong assumption: additivity.

**QMIX:**
```
Q_tot = f_mix(Q₁(o₁,a₁), ..., Qₙ(oₙ,aₙ); s)
```
Mixing network combines individual Qs with weights conditioned on global state s.

**IGM condition (Individual-Global Max):**
```
∂Q_tot / ∂Qᵢ ≥ 0    for all i
```
Monotonicity constraint ensures argmax of Q_tot equals individual argmaxes → allows decentralized greedy execution. QMIX is more expressive than VDN; still allows decentralized execution.

### Centralized Critic Methods

**COMA (Counterfactual Multi-Agent):**
- Centralized critic estimates Q(s,**a**)
- **Counterfactual baseline** for credit assignment:
```
Aᵢ(s,a) = Q(s,a) − Σ_{a'ᵢ} πᵢ(a'ᵢ|oᵢ) Q(s, a₋ᵢ, a'ᵢ)
```
Isolates agent i's contribution by marginalizing over its own actions while holding others fixed.

**MADDPG (Multi-Agent DDPG):**
- Centralized critic Qᵢ(s, a₁,...,aₙ) per agent
- Decentralized actor μᵢ(oᵢ)
- Off-policy with replay buffer
- Works for cooperative and competitive settings

**MAPPO (Multi-Agent PPO):**
- PPO with shared centralized critic (takes global state)
- Each agent has own actor (or shared via parameter sharing)
- Strong cooperative baseline; surprisingly effective

### Additional Techniques

**Self-Play:** Train against copies of itself → progressively harder opponents. Risk: cyclic strategies (rock-paper-scissors dynamics).

**Parameter Sharing:** All agents share policy network weights; agent identity encoded as input. Valid only for homogeneous agents — reduces sample complexity.

**Communication:** Agents exchange messages (DIAL, CommNet). Differentiable communication learns *what* to communicate.

### MARL Algorithm Summary
| Algorithm | Setting | Approach | Key Property |
|-----------|---------|----------|--------------|
| IQL | Coop/Comp | Off-policy, independent | Simple; non-stationarity problem |
| VDN | Cooperative | Value factorization | Additive Q decomposition |
| QMIX | Cooperative | Value factorization | Monotonic mixing; IGM condition |
| COMA | Cooperative | Centralized critic | Counterfactual advantage |
| MADDPG | Mixed/Competitive | Centralized critic + DDPG | Continuous actions |
| MAPPO | Cooperative | Centralized critic + PPO | Strong baseline |

---

## Cross-Module Comparison Table

| Algorithm | Module | On/Off | Model-free | Discrete | Continuous | Multi-agent |
|-----------|--------|--------|-----------|----------|------------|-------------|
| Q-learning/SARSA | 1 | On/Off | Yes | Yes | No (tabular) | No |
| DQN | 2 | Off | Yes | Yes | No | No |
| DDQN | 3 | Off | Yes | Yes | No | No |
| REINFORCE | 3 | On | Yes | Yes | Yes | No |
| Actor-Critic | 3 | On | Yes | Yes | Yes | No |
| DDPG | 4 | Off | Yes | No | Yes | MADDPG |
| PPO | 4 | On | Yes | Yes | Yes | MAPPO |
| QMIX/VDN | 5 | Off | Yes | Yes | No | Yes (coop) |
| COMA | 5 | On | Yes | Yes | No | Yes (coop) |

---

## Exam Question Patterns

### "Explain X concept"
- MDP components: always name all 5: (S, A, P, R, γ) and what each means
- Markov property: current state sufficient — no need for history
- Bellman: always state whether it's expectation (evaluates π) or optimality (finds π*)
- CTDE: mention both phases and why each is needed

### "Compare X and Y"
- SARSA vs Q-learning: on- vs off-policy, safe vs optimal path on CliffWalking
- MC vs TD: bias/variance, model requirements, episodic requirement
- DQN vs DDQN: same architecture, just target formula — DDQN reduces overestimation
- DDPG vs PPO: deterministic vs stochastic, off vs on policy, replay vs rollout
- VDN vs QMIX: additive vs monotonic mixing, QMIX has global state conditioning

### "What problem does X solve?"
- Replay buffer → correlated samples
- Target network → moving target (non-stationary Q target)
- DDQN → overestimation bias in DQN
- Baseline in REINFORCE → variance reduction (unbiased)
- GAE → bias/variance tradeoff in advantage estimation
- CTDE → non-stationarity + credit assignment while enabling scalable execution
- IGM in QMIX → guarantees individual greedy actions = joint greedy action

### "Derive / write the update rule"
- Know all update rules from memory: TD(0), SARSA, Q-learning, DQN target, DDQN target, policy gradient, GAE
- For actor-critic: actor loss = `-log π(a|s) · δ`; critic loss = `(V(s) − y)²`
- For PPO: clipped ratio with advantage; know what happens when Â > 0 vs Â < 0

### "Why does algorithm X converge / not converge?"
- Q-learning: converges to Q* under GLIE + Robbins-Monro step sizes
- SARSA: converges to optimal ε-greedy policy under GLIE
- Independent learners in MARL: non-stationarity breaks convergence
- DQN without target net: moving target can oscillate/diverge

### Key Numerical Facts
- Toy MDP (exercises): Q*(A,0)≈2.8; Q*(B,1)=2; Vπ(A)≈1.883, Vπ(B)≈1.878
- PPO clip ε = 0.2 (typical); GAE λ = 0.95 (typical)
- DDPG soft update τ = 0.005 (typical)
- DQN CartPole target: eval return ~400–500 with full DQN
