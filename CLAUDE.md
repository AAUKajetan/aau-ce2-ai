# AAU CE2 AI Course — Study Materials Map

## Teach-Back Session Rules

- We are preparing for the exam using teach-back style: Claude asks questions, the student answers
- **Always assume the student's answer may be incomplete or incorrect — check every answer against the course materials and correct any mistakes, even small ones**
- Always give feedback structured as: what was correct, what was wrong or missing, the precise correction
- Never skip corrections to be polite — catching errors now is the point

## Repo Structure

| Path | What's there |
|------|--------------|
| `exam_prep/` | **Study core.** Lecture MDs, exercise MDs, `Overview_*.md`, `Models_Table.md`, `Cheatsheet_ML_DNN.html`, `Cheatsheet_RL.html`, plus `figures/`, `slides/`, `exercises/`. Used for teach-back sessions. |
| `code/autoencoder/` | MNIST autoencoder experiment (Module 9) + `outputs/` plots + MNIST `data/`. |
| `code/miniproject/` | Rice classification mini-project: CNN vs ViT. |
| `code/class-exercises/` | Raw in-class notebooks/PDFs (`module1`–`module4`). |
| `code/scratch/` | Throwaway notebooks (`testbook.ipynb`). |
| `tools/` | `clean_pdf_md.py` — PDF→Markdown cleaner used to build the notes. |

Note: lecture MDs and `Cheatsheet_ML_DNN.html` reference images as
`figures/<name>.png`. Those files must stay alongside `exam_prep/figures/`
or the relative links break.

## exam_prep/ Structure

### Lecture Notes (slides → MD)

| Module | Topic | File |
|--------|-------|------|
| Module 1 | AI/ML taxonomy, preprocessing, PCA, LDA | `exam_prep/Module_1.md` |
| Module 2 | MLP, backprop, CNN, regularization | `exam_prep/Module_2.md` |
| Module 3 | RNN, LSTM, GRU, embeddings | `exam_prep/Module_3.md` |
| RL Module 1 | MDP, Bellman, DP, MC, TD, SARSA, Q-learning | `exam_prep/RL_Lecture_Module1.md` |
| RL Lecture 2 | DQN, replay buffer, target network | `exam_prep/RL_Lecture2.md` |
| DRL Module 3 | DDQN, policy gradients, REINFORCE, actor-critic | `exam_prep/DRL_Module3.md` |
| RL Lecture 4 | DDPG, PPO, GAE, continuous control | `exam_prep/RL_Lecture4.md` |
| RL Lecture 5 | MARL, CTDE, VDN, QMIX, COMA, MADDPG, MAPPO | `exam_prep/RL_Lecture5.md` |
| Module 9 | Autoencoders, VAE, ELBO, reparameterization | `exam_prep/Module_9.md` |
| Module 10 | Attention, Transformers, GANs | `exam_prep/Module_10.md` |

---

### Exercise Notes (exam_prep/exercises/ → MD)

The source exercise PDFs/notebooks live in `exam_prep/exercises/`; the worked
write-ups are the Notes MDs below (in `exam_prep/`).

| Exercise File | Topic | Notes MD | Lecture MD |
|---------------|-------|----------|------------|
| `Exercises_Module_1.pdf` | AI/ML concepts, PCA & LDA on Iris | `exam_prep/Exercises_Module_1.md` | `Module_1.md` |
| `Exercises (1).pdf` | MLP backprop, CNN design, MNIST | `exam_prep/Exercises_Module_2.md` | `Module_2.md` |
| `Exercises_Module_3.pdf` | RNN/LSTM theory & coding | `exam_prep/Exercises_Module_3.md` | `Module_3.md` |
| `Exercises (2).pdf` | RL Module 1: Toy MDP, MC, Bellman, Q-table | `exam_prep/Exercises_RL_Module1.md` | `RL_Lecture_Module1.md` |
| `Excercises-RL_Module2.pdf` | SARSA vs Q-learning, DQN ablations on CartPole | `exam_prep/Exercises_RL_Module2.md` | `RL_Lecture2.md` |
| `Module3_Exercise_Set.pdf` | DQN, DDQN, REINFORCE, Actor-Critic | `exam_prep/Exercises_DRL_Module3.md` | `DRL_Module3.md` |
| `Module4_Python_Tutorial (2).pdf` | DDPG & PPO on Pendulum | `exam_prep/Exercises_RL_Module4.md` | `RL_Lecture4.md` |
| `Exercises_Module_4.pdf` | Autoencoders, VAE, reparameterization | `exam_prep/Exercises_Module_4.md` | `Module_9.md` |
| `Exercises_Module_5.pdf` | GANs, attention, MHA from scratch | `exam_prep/Exercises_Module_5.md` | `Module_10.md` |

---

## Quick Reference

- **RL environments used in exercises:** CliffWalking-v0 (tabular), CartPole-v1 (DQN/REINFORCE/AC), Pendulum-v1 (DDPG/PPO)
- **Install (RL):** `pip install gymnasium[classic-control] gymnasium[toy-text] torch numpy matplotlib`
- **Install (DNN/autoencoder):** `pip install -r code/autoencoder/requirements.txt`
- **Key formulas file:** each lecture MD contains the exact equations; each exercise MD has worked answers and code patterns
- **Runnable code:** lives under `code/` (each subfolder has its own README); utility scripts under `tools/`
