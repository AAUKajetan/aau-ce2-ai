# AAU CE2 — AI & Advanced Machine Learning

A personal knowledge base for the AAU CE2 "AI and Advanced Machine Learning"
course, covering both **deep learning (ML/DNN)** and **reinforcement learning (RL)**.
It holds two things: polished **study notes** for exam prep, and the **code
experiments** done during the course.

## Repository layout

```
aau-ce2-ai/
├── exam_prep/        ← STUDY CORE: lecture notes, exercise notes, overviews, cheatsheets, figures, slides
├── code/             ← Code experiments (autoencoder, mini-project, in-class exercises, scratch)
│   ├── autoencoder/      MNIST autoencoder (Module 9) + generated outputs/
│   ├── miniproject/      Rice classification: CNN vs ViT
│   ├── class-exercises/  Raw in-class notebooks/PDFs (module1–module4)
│   └── scratch/          Throwaway notebooks
├── tools/            ← Utility scripts (PDF→Markdown cleaner)
├── CLAUDE.md         ← Rules for AI-assisted teach-back study sessions + file map
└── README.md         ← This file
```

The directory layout (study vs. code) is deliberate: `exam_prep/` is what you
open to revise; `code/` is what you open to run things.

## How to use this for exam prep

Work from the general to the specific:

1. **Overviews** — start with the per-class exam overviews to see the whole map:
   - `exam_prep/Overview_ML_Class.md`
   - `exam_prep/Overview_DNN_Class.md`
   - `exam_prep/Overview_RL_Class.md`
   - `exam_prep/Models_Table.md` — one-glance table of every model/method.
2. **Lecture notes** — drill into a topic with the per-module lecture MDs
   (`exam_prep/Module_*.md`, `exam_prep/RL_Lecture*.md`, `exam_prep/DRL_Module3.md`).
   These contain the exact formulas and embedded figures.
3. **Exercise notes** — practice with the worked exercise write-ups
   (`exam_prep/Exercises_*.md`), each paired to its lecture module.
4. **Cheatsheets** — final-stretch review, open in a browser:
   - `exam_prep/Cheatsheet_ML_DNN.html`
   - `exam_prep/Cheatsheet_RL.html`

The exact module → file mapping is in [`CLAUDE.md`](CLAUDE.md).

### Teach-back study sessions

`CLAUDE.md` defines a teach-back protocol: an AI assistant asks questions, you
answer, and it checks every answer against these materials and corrects mistakes.
Open the repo with your assistant and ask it to quiz you on a module.

## Running the code

Each `code/` subfolder has its own README. Quick pointers:

- **Autoencoder:** `cd code/autoencoder && pip install -r requirements.txt && python mnist_autoencoder.py`
- **Mini-project:** see `code/miniproject/README.md` (full pipeline + dataset download).
- **RL exercises:** `pip install gymnasium[classic-control] gymnasium[toy-text] torch numpy matplotlib`

## Notes

- `exam_prep/figures/` images are referenced by the lecture MDs and the ML/DNN
  cheatsheet as `figures/<name>.png`. Keep these files together — moving an MD or
  the cheatsheet away from `figures/` breaks those relative links.
- `exam_prep/figures.zip` is a stale partial backup of the figures folder and can
  be ignored; the live `figures/` folder is the source of truth.
