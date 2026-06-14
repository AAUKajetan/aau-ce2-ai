# Exercises — RL Module 1: Foundations (Toy MDP)

## Shared Toy MDP

States: {A, B, T} where T is terminal. Actions: {0, 1}. γ = 0.9

| State | Action | Next | Reward |
|-------|--------|------|--------|
| A | 0 | B | +1 |
| A | 1 | T | 0 |
| B | 0 | A | 0 |
| B | 1 | T | +2 |

Stochastic policy π: π(0|A)=0.7, π(1|A)=0.3, π(0|B)=0.4, π(1|B)=0.6

Return: G₀ = Σₖ γᵏ rₖ₊₁

---

## Q1: Monte Carlo — Simulate Episodes

**Goal:** Practice generating trajectories and computing discounted returns.

**Tasks:**
1. Implement `step(state, action) → (next_state, reward, done)`
2. Implement stochastic policy sampling
3. Simulate N=2000 episodes from s₀=A
4. Compute G₀ per episode, report mean and std

**Key insight:** Store each transition as (s, a, r, s', done) for reuse in later questions.

**Answer hints — typical results:**
- Main paths from A: A→B→T (reward=1+γ·2=2.8) or A→T (reward=0) or A→B→A→... (longer)
- Expected return should be positive (B→T gives +2 which is high)

---

## Q2: Policy Evaluation via Bellman Expectation

**Goal:** Solve for Vπ(A) and Vπ(B) analytically or iteratively.

**Bellman equations** (with Vπ(T)=0):
```
Vπ(A) = 0.7·(1 + γ·Vπ(B)) + 0.3·(0 + γ·0)
       = 0.7 + 0.63·Vπ(B)

Vπ(B) = 0.4·(0 + γ·Vπ(A)) + 0.6·(2 + γ·0)
       = 0.36·Vπ(A) + 1.2
```

**Solving algebraically:**
- Substitute: Vπ(A) = 0.7 + 0.63·(0.36·Vπ(A) + 1.2)
- Vπ(A) = 0.7 + 0.2268·Vπ(A) + 0.756
- 0.7732·Vπ(A) = 1.456 → **Vπ(A) ≈ 1.883**
- Vπ(B) = 0.36·1.883 + 1.2 = **1.878**

**Verify:** MC mean from Q1 should be close to Vπ(A) ≈ 1.883

**Iterative approach:** Initialize V={A:0,B:0}, repeatedly apply Bellman backup until Δ < 1e-10.

---

## Q3: Bellman Optimality Backup + ε-greedy

**Goal:** Implement Q-value iteration and understand exploration.

**Bellman optimality backup:**
```
Q_new(s,a) ← r + γ · max_a' Q(s', a')
```

**Steps:**
1. Initialize Q[(s,a)] = 0 for all s,a
2. Perform K synchronous sweeps (compute Q_new from Q then replace)
3. After each sweep print greedy actions and Q-values

**ε-greedy:** with prob ε choose random action; else argmax Q with random tie-breaking

**Expected convergence (after 6 sweeps):**
- Q*(B,1) should be highest: r=2, terminal → Q*(B,1)=2
- Q*(A,0) → B → take action 1: 1 + γ·2 = 2.8 → **Q*(A,0) ≈ 2.8**
- Q*(A,1) = 0 (goes to terminal with 0 reward)
- Greedy: A→action 0, B→action 1

---

## Short Answer Questions

**1. What does γ do conceptually?**
Controls how much future rewards are discounted vs immediate rewards. γ=0: myopic (only immediate reward). γ→1: far-sighted (future rewards matter equally). Prevents infinite returns in continuing tasks.

**2. Why do we need exploration?**
A greedy policy only exploits current knowledge. Unexplored states/actions may have higher value. Without exploration, the agent may converge to a suboptimal policy.

**3. Bellman expectation vs optimality?**
- **Expectation:** evaluates a given policy π: Vπ(s) = Σ_a π(a|s) [r + γ Vπ(s')]
- **Optimality:** finds the best policy: V*(s) = max_a [r + γ V*(s')]
Expectation asks "how good is this policy?"; optimality asks "what is the best possible?"
