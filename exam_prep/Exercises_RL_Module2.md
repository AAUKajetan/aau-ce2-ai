# Exercises — RL Module 2: Tabular SARSA, Q-learning, and DQN

## Setup

```bash
pip install gymnasium torch numpy matplotlib
pip install "gymnasium[classic-control]"  # CartPole
pip install "gymnasium[toy-text]"          # CliffWalking
```

---

## Exercise 1: SARSA vs Q-learning on CliffWalking

**Environment:** CliffWalking-v0 — 4×12 grid, 48 states, 4 actions (UP/RIGHT/DOWN/LEFT)
- Reward: -1/step; cliff: -100 and reset; episode ends at goal

**Update rules:**
```
SARSA (on-policy):    Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
Q-learning (off-pol): Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

**Key difference:** SARSA uses a' sampled from the ε-greedy policy; Q-learning uses max — it optimizes the greedy policy even while exploring.

**Hyperparameters:** α=0.1, γ=0.99, ε decay 1.0→0.05 over 3000 eps, 10000 total episodes

**What to implement:**
1. Q-table Q[s,a] initialized to zeros
2. `eps_greedy(Q, s, eps, rng)` with random tie-breaking
3. `train_sarsa(...)` and `train_qlearning(...)`
4. Log episode returns and cliff falls (r ≤ -100)

**What to plot:** Episode return (raw + moving avg 200) and cliff falls per 100 episodes

**Expected result:**
- Q-learning converges to optimal (shorter) path but falls more often during training (because it optimizes greedy policy while exploring dangerously)
- SARSA converges to safer path (stays away from cliff) because it accounts for ε-greedy exploration in its updates

**On-policy vs off-policy risk:** SARSA "knows" it might slip off due to ε-random actions → learns cautious path. Q-learning ignores exploration policy → learns cliff-edge optimal path.

---

## Exercise 2: On-policy vs Off-policy Visibility

**Goal:** Separate training return (exploration on) from evaluation return (greedy, ε=0)

**Protocol:**
- Every K=200 episodes, evaluate greedy policy for M=20 episodes (no learning, ε=0)
- Plot training return AND evaluation return per episode for both methods

**Key questions:**
- Training return is noisier/worse because ε-random actions sometimes fall off cliff
- SARSA's greedy evaluation may lag if ε stays fixed (on-policy Q value converges to the ε-greedy return, not the greedy one)
- Q-learning's greedy evaluation often converges faster — its Q values already target the greedy policy

**Important rules:** During evaluation: no Q updates, ε=0, fixed seed for reproducibility.

---

## Exercise 3: DQN on CartPole

**Environment:** CartPole-v1 — 4D continuous state, 2 discrete actions, reward +1/step, episode ends when pole falls or 500 steps

**DQN target (with terminal masking):**
```
y = r + (1 - done) · γ · max_a' Q_{θ⁻}(s', a')
```

**Architecture:** QNet: Linear(4→128)→ReLU→Linear(128→128)→ReLU→Linear(128→2)

**Key components:**
- `ReplayBuffer(capacity=100_000)`: deque storing (s,a,r,s',done)
- Online net Qθ + Target net Qθ⁻ (hard copy every C=1000 steps)
- ε-decay: 1.0→0.05 over 50_000 steps

**Huber loss (SmoothL1):** Quadratic for |e|≤δ, linear for |e|>δ — robust to large TD errors

**Baseline hyperparameters:** total_steps=200_000, batch=64, γ=0.99, lr=1e-3, Adam

**Ablation matrix:**
| Condition | Replay | Target | Loss | Optimizer |
|-----------|--------|--------|------|-----------|
| Baseline | ON | ON | Huber | Adam |
| A1: No target | ON | OFF | Huber | Adam |
| A2: No replay | OFF | ON | Huber | Adam |
| A3: No Adam | ON | ON | Huber | SGD |
| A4: MSE | ON | ON | MSE | Adam |

**Debug checklist:**
- Terminal masking: if done=True, y = r (no bootstrap)
- Tensor shapes: qnet(s) → [B, A]; gather → [B, 1]
- Warm up replay 1000 steps before training
- Evaluate with ε=0, no network updates

**Adam vs SGD:** Adam adapts step size per parameter (helpful for noisy RL gradients). SGD needs larger lr tuning; may be slower or unstable with same lr as Adam.

**Expected baseline:** CartPole eval return should approach 400–500 with full DQN.

**Bonus:** Implement DDQN — select next action with online net, evaluate with target net:
```python
a_star = qnet(s2).argmax(dim=1, keepdim=True)
y = r + (1-done) * gamma * target_qnet(s2).gather(1, a_star)
```
