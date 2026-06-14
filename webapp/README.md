# AAU CE2 AI Study Hub — Webapp

A fully static, zero-dependency study webapp for the AAU CE2 AI/ML/DNN/RL exam.
Renders all lecture notes, exercise notes, and cheat sheets as a navigable single-page app,
with full-screen figure lightbox and responsive dark/light mode.

---

## Requirements

- **Node.js 16+** — nothing else. The `marked` renderer is vendored in `webapp/vendor/marked`,
  so the build works on any machine without `npm install`. (If `marked` is installed via
  node_modules it takes precedence.)
- Or just **Docker** (see below).

---

## Build

From the repo root:

```sh
node webapp/build.js
```

Or equivalently, from inside `webapp/`:

```sh
npm run build
```

This:
1. Reads all markdown files from `exam_prep/`
2. Renders them to HTML using `marked` (GFM mode — tables, code fences, etc.)
3. Embeds the cheat sheet HTML pages directly
4. Copies all 80 figures from `exam_prep/figures/` into `dist/figures/`
5. Writes a single self-contained `dist/index.html`

Build output: `webapp/dist/` (~210 KB HTML + ~11 MB figures)

---

## Run on port 6767

### Option A — Docker (recommended for the private server)

From the **repo root** (the image build needs `exam_prep/` as content source):

```sh
docker compose up -d --build
# or without compose:
docker build -f webapp/Dockerfile -t studyhub .
docker run -d --name studyhub -p 6767:6767 --restart unless-stopped studyhub
```

`restart: unless-stopped` keeps it alive across reboots. To pick up content changes
in `exam_prep/`, rebuild: `docker compose up -d --build`.

### Option B — Built-in Node server (no install needed)

```sh
node webapp/server.js
```

Then open: **http://localhost:6767** or **http://<server-ip>:6767** from any device on the LAN.

The server binds to `0.0.0.0` so it is reachable over the local network.

Override the port if needed:
```sh
PORT=8080 node webapp/server.js
```

### Option B — npm run scripts

```sh
# Build only
npm run build

# Serve only (assumes dist/ already built)
npm run preview

# Build + serve in one command
npm run start
```

### Option C — npx serve (if available)

```sh
npx serve dist -l 6767 -s
```

The `-s` flag enables SPA fallback (serves index.html for unknown paths, needed for hash routing).

### Option D — Python fallback

```sh
cd webapp/dist && python3 -m http.server 6767 --bind 0.0.0.0
```

Note: Python's server does not support SPA fallback, but since all navigation is hash-based (`#page-id`) this works fine.

---

## Refreshing figures after exam_prep changes

If you add or update figures in `exam_prep/figures/`, run:

```sh
# Rebuild everything (safest):
node webapp/build.js

# Or copy figures only (if dist/ already exists):
npm run refresh-figures
```

---

## Content map

| Nav Group    | Pages included |
|-------------|----------------|
| Overviews   | ML Class Overview, DNN Class Overview, RL Class Overview |
| ML Class    | Module 1 — ML Foundations |
| DNN Class   | Module 2 (MLP/CNN), Module 3 (RNN/LSTM), Module 9 (AE/VAE), Module 10 (Attention/GANs) |
| RL Class    | RL 1–5 (MDP → DQN → Policy Gradients → DDPG/PPO → MARL) |
| Exercises   | 9 exercise sets (ex-ml1 through ex-gan) |
| Cheat Sheets| ML & DNN cheat sheet, RL cheat sheet, Models table |

---

## Features

- **Hash routing** — each page has a stable URL (`#module2`, `#rl3`, etc.)
- **Figure lightbox** — click any diagram to enlarge; click outside or press Esc to close
- **Dark/light mode** — follows `prefers-color-scheme` automatically
- **Responsive** — hamburger menu on mobile/narrow viewports
- **GFM tables** — all comparison tables render correctly
- **Code blocks** — syntax-highlighted-ready monospace with horizontal scroll

---

## File structure

```
webapp/
├── build.js          # Build script (Node)
├── server.js         # Static file server on port 6767
├── Dockerfile        # Container build (run from repo root context)
├── package.json      # npm scripts wrapper
├── vendor/marked/    # Vendored markdown renderer (no npm install needed)
├── README.md         # This file
└── dist/             # Build output (generated, not committed)
    ├── index.html    # Complete SPA (~210 KB, all content embedded)
    └── figures/      # 80 PNG diagrams copied from exam_prep/figures/
```

`dist/` is generated at build time and already covered by the repo `.gitignore`
(`dist/` pattern). `docker-compose.yml` lives at the repo root.
