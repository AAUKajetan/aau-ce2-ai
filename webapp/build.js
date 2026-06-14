#!/usr/bin/env node
/**
 * Build script for AAU CE2 AI Study Webapp
 * Uses 'marked' (vendored in webapp/vendor/marked, or node_modules if installed)
 * Output: webapp/dist/  (fully static, no server-side dependencies)
 */

const fs = require('fs');
const path = require('path');

function loadMarked() {
  const candidates = [
    'marked',                                  // node_modules / global resolution
    path.join(__dirname, 'vendor', 'marked'),  // vendored copy (always present)
  ];
  for (const c of candidates) {
    try { return require(c); } catch (e) { /* try next */ }
  }
  console.error('FATAL: could not load "marked". Run `npm install marked` in webapp/ or restore webapp/vendor/marked.');
  process.exit(1);
}
const { marked } = loadMarked();

// ── Configure marked ──────────────────────────────────────────────────────────
marked.setOptions({ gfm: true, breaks: false });

// ── Paths ─────────────────────────────────────────────────────────────────────
const EXAM_PREP = path.resolve(__dirname, '../exam_prep');
const DIST      = path.resolve(__dirname, 'dist');
const FIGURES_SRC = path.join(EXAM_PREP, 'figures');
const FIGURES_DST = path.join(DIST, 'figures');

// ── Navigation structure ──────────────────────────────────────────────────────
const NAV = [
  {
    group: 'Overviews',
    icon: '📋',
    items: [
      { id: 'overview-ml',  label: 'ML Class Overview',  file: 'Overview_ML_Class.md' },
      { id: 'overview-dnn', label: 'DNN Class Overview', file: 'Overview_DNN_Class.md' },
      { id: 'overview-rl',  label: 'RL Class Overview',  file: 'Overview_RL_Class.md' },
    ]
  },
  {
    group: 'ML Class',
    icon: '🤖',
    items: [
      { id: 'module1',  label: 'Module 1 — ML Foundations',   file: 'Module_1.md' },
    ]
  },
  {
    group: 'DNN Class',
    icon: '🧠',
    items: [
      { id: 'module2',  label: 'Module 2 — MLP & CNN',        file: 'Module_2.md' },
      { id: 'module3',  label: 'Module 3 — RNN & LSTM',       file: 'Module_3.md' },
      { id: 'module9',  label: 'Module 9 — Autoencoders & VAE', file: 'Module_9.md' },
      { id: 'module10', label: 'Module 10 — Attention & GANs', file: 'Module_10.md' },
    ]
  },
  {
    group: 'RL Class',
    icon: '🎮',
    items: [
      { id: 'rl1', label: 'RL 1 — MDP, Bellman, DP, MC, TD', file: 'RL_Lecture_Module1.md' },
      { id: 'rl2', label: 'RL 2 — DQN',                      file: 'RL_Lecture2.md' },
      { id: 'rl3', label: 'RL 3 — DDQN, Policy Gradients',   file: 'DRL_Module3.md' },
      { id: 'rl4', label: 'RL 4 — DDPG, PPO',                file: 'RL_Lecture4.md' },
      { id: 'rl5', label: 'RL 5 — MARL, QMIX, MADDPG',       file: 'RL_Lecture5.md' },
    ]
  },
  {
    group: 'Exercises',
    icon: '📝',
    items: [
      { id: 'ex-ml1',  label: 'Ex 1 — ML Concepts & PCA',     file: 'Exercises_Module_1.md' },
      { id: 'ex-ml2',  label: 'Ex 2 — MLP & CNN',             file: 'Exercises_Module_2.md' },
      { id: 'ex-ml3',  label: 'Ex 3 — RNN & LSTM',            file: 'Exercises_Module_3.md' },
      { id: 'ex-rl1',  label: 'Ex 4 — Toy MDP, MC, Bellman',  file: 'Exercises_RL_Module1.md' },
      { id: 'ex-rl2',  label: 'Ex 5 — SARSA, Q-learning, DQN',file: 'Exercises_RL_Module2.md' },
      { id: 'ex-rl3',  label: 'Ex 6 — DQN/DDQN/REINFORCE/AC', file: 'Exercises_DRL_Module3.md' },
      { id: 'ex-rl4',  label: 'Ex 7 — DDPG & PPO',            file: 'Exercises_RL_Module4.md' },
      { id: 'ex-ae',   label: 'Ex 8 — Autoencoders & VAE',     file: 'Exercises_Module_4.md' },
      { id: 'ex-gan',  label: 'Ex 9 — GANs, Attention, MHA',  file: 'Exercises_Module_5.md' },
    ]
  },
  {
    group: 'Cheat Sheets',
    icon: '📌',
    items: [
      { id: 'cheat-ml',  label: 'ML & DNN Cheat Sheet', file: 'Cheatsheet_ML_DNN.html',  isHtml: true },
      { id: 'cheat-rl',  label: 'RL Cheat Sheet',       file: 'Cheatsheet_RL.html',      isHtml: true },
      { id: 'models',    label: 'Models Table',          file: 'Models_Table.md' },
    ]
  },
];

// ── Helpers ───────────────────────────────────────────────────────────────────

function readFile(filePath) {
  try { return fs.readFileSync(filePath, 'utf8'); }
  catch (e) { return null; }
}

/** Rewrite `figures/foo.png` → `figures/foo.png` (already correct relative path from dist root).
    MD files use relative path `figures/foo.png`; we serve dist/ so that resolves correctly. */
function renderMd(mdContent) {
  // figures refs are already `figures/xxx.png` — correct for dist/
  return marked.parse(mdContent);
}

/**
 * Extract <body> content from an HTML file, PRESERVING its <style> rules.
 * Styles from <head> are kept but scoped under `.cheatsheet-embed` so the
 * cheat sheet's own `body { … }` rules don't leak into the app shell.
 */
function extractHtmlBody(htmlContent) {
  // Collect <style> blocks (typically inside <head>) before stripping it
  const styles = [];
  const styleRe = /<style[^>]*>([\s\S]*?)<\/style>/gi;
  let m;
  while ((m = styleRe.exec(htmlContent)) !== null) styles.push(m[1]);

  let body = htmlContent
    .replace(/<!DOCTYPE[^>]*>/i, '')
    .replace(/<html[^>]*>/i, '')
    .replace(/<\/html>/i, '')
    .replace(/<head[\s\S]*?<\/head>/i, '')
    .replace(/<body[^>]*>/i, '')
    .replace(/<\/body>/i, '')
    .trim();

  let scopedCss = '';
  if (styles.length) {
    scopedCss = styles.join('\n')
      // `body { … }` → scope to the embed wrapper
      .replace(/(^|\})\s*body\s*\{/g, '$1 .cheatsheet-embed {')
      // scope bare element/class selectors by prefixing (cheap but effective:
      // prefix every top-level selector that isn't an at-rule)
      .split('\n').map(line => {
        const t = line.trim();
        if (/^(@|\}|\/|$)/.test(t) || !/\{\s*$|\{.*\}/.test(t)) return line;
        if (t.startsWith('.cheatsheet-embed')) return line;
        return line.replace(/^(\s*)([^@{}]+)\{/, (_, ws, sel) => {
          const scoped = sel.split(',').map(s => {
            s = s.trim();
            return s.startsWith('.cheatsheet-embed') ? s : `.cheatsheet-embed ${s}`;
          }).join(', ');
          return `${ws}${scoped} {`;
        });
      }).join('\n');
    scopedCss = `<style>\n${scopedCss}\n</style>\n`;
  }
  return scopedCss + body;
}

/** Build the sidebar HTML */
function buildSidebar(allItems) {
  let html = '';
  for (const group of NAV) {
    html += `<div class="nav-group">
  <div class="nav-group-label">${group.icon} ${group.group}</div>
  <ul>`;
    for (const item of group.items) {
      html += `\n    <li><a href="#${item.id}" class="nav-link" data-page="${item.id}">${item.label}</a></li>`;
    }
    html += `\n  </ul>\n</div>\n`;
  }
  return html;
}

/** Build JS page data object */
function buildPageData(allItems) {
  const pages = {};
  for (const item of allItems) {
    const srcPath = path.join(EXAM_PREP, item.file);
    const content = readFile(srcPath);
    if (!content) {
      console.warn(`  WARNING: missing file ${item.file}`);
      pages[item.id] = `<p class="missing">File not found: ${item.file}</p>`;
      continue;
    }
    if (item.isHtml) {
      // For cheat sheets: embed their (scoped) <style> and body content
      pages[item.id] = `<div class="cheatsheet-embed">${extractHtmlBody(content)}</div>`;
    } else {
      pages[item.id] = renderMd(content);
    }
  }
  return pages;
}

// ── Main ──────────────────────────────────────────────────────────────────────

console.log('Building AAU CE2 AI Study Webapp...');

// Create dist directory
fs.mkdirSync(DIST, { recursive: true });

// Copy figures
if (fs.existsSync(FIGURES_SRC)) {
  fs.mkdirSync(FIGURES_DST, { recursive: true });
  const figures = fs.readdirSync(FIGURES_SRC);
  let copied = 0;
  for (const fig of figures) {
    if (fig.match(/\.(png|jpg|jpeg|gif|svg|webp)$/i)) {
      fs.copyFileSync(path.join(FIGURES_SRC, fig), path.join(FIGURES_DST, fig));
      copied++;
    }
  }
  console.log(`  Copied ${copied} figures to dist/figures/`);
} else {
  console.warn('  WARNING: figures directory not found');
}

// Collect all items
const allItems = NAV.flatMap(g => g.items);
const firstPageId = allItems[0].id;

// Render all content
console.log('  Rendering markdown and HTML content...');
const pages = buildPageData(allItems);

// Serialize page content as JSON for embedding in the HTML
// We escape </script> occurrences inside the JSON to avoid breaking the script tag
const pagesJson = JSON.stringify(pages).replace(/<\/script>/gi, '<\\/script>');

// Sidebar HTML
const sidebarHtml = buildSidebar(allItems);

// ── Generate index.html ───────────────────────────────────────────────────────
const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AAU CE2 — AI/ML/DNN/RL Study Hub</title>
  <style>
    /* ── Reset & base ─────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #f8f9fa;
      --surface:   #ffffff;
      --border:    #e2e8f0;
      --text:      #1a202c;
      --text-muted:#64748b;
      --accent:    #4f46e5;
      --accent-bg: #eef2ff;
      --code-bg:   #f1f5f9;
      --nav-width: 270px;
      --header-h:  52px;
      --shadow:    0 1px 3px rgba(0,0,0,.08);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg:        #0f172a;
        --surface:   #1e293b;
        --border:    #334155;
        --text:      #e2e8f0;
        --text-muted:#94a3b8;
        --accent:    #818cf8;
        --accent-bg: #1e1b4b;
        --code-bg:   #0f172a;
        --shadow:    0 1px 3px rgba(0,0,0,.4);
      }
    }

    html, body { height: 100%; }
    body {
      font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 15px;
      line-height: 1.6;
    }

    /* ── Layout ───────────────────────────────────── */
    #app { display: flex; flex-direction: column; height: 100vh; }

    #header {
      height: var(--header-h);
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      padding: 0 1rem;
      gap: 0.75rem;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: var(--shadow);
      flex-shrink: 0;
    }
    #header h1 {
      font-size: 1rem;
      font-weight: 600;
      color: var(--accent);
      letter-spacing: -.01em;
    }
    #header .badge {
      font-size: 0.7rem;
      background: var(--accent-bg);
      color: var(--accent);
      padding: 2px 8px;
      border-radius: 999px;
      font-weight: 500;
    }
    #menu-btn {
      display: none;
      background: none;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 4px 8px;
      cursor: pointer;
      color: var(--text);
      font-size: 1.1rem;
      margin-right: 0.25rem;
    }

    #body { display: flex; flex: 1; overflow: hidden; }

    /* ── Sidebar ──────────────────────────────────── */
    #sidebar {
      width: var(--nav-width);
      background: var(--surface);
      border-right: 1px solid var(--border);
      overflow-y: auto;
      flex-shrink: 0;
      padding: 0.75rem 0;
    }
    #sidebar-overlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,.4);
      z-index: 90;
    }

    .nav-group { margin-bottom: 0.25rem; }
    .nav-group-label {
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: .06em;
      text-transform: uppercase;
      color: var(--text-muted);
      padding: 0.5rem 1rem 0.25rem;
    }
    .nav-group ul { list-style: none; }
    .nav-link {
      display: block;
      padding: 0.3rem 1rem 0.3rem 1.25rem;
      font-size: 0.82rem;
      color: var(--text);
      text-decoration: none;
      border-radius: 0;
      transition: background .12s;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .nav-link:hover { background: var(--accent-bg); color: var(--accent); }
    .nav-link.active {
      background: var(--accent-bg);
      color: var(--accent);
      font-weight: 600;
      border-right: 3px solid var(--accent);
    }

    /* ── Main content ─────────────────────────────── */
    #main {
      flex: 1;
      overflow-y: auto;
      padding: 2rem 2.5rem;
    }
    #content { max-width: 860px; margin: 0 auto; }

    /* ── Markdown styles ──────────────────────────── */
    #content h1 { font-size: 1.7rem; font-weight: 700; margin: 0 0 1rem; color: var(--text); border-bottom: 2px solid var(--accent); padding-bottom: .4rem; }
    #content h2 { font-size: 1.25rem; font-weight: 600; margin: 1.8rem 0 .6rem; color: var(--text); }
    #content h3 { font-size: 1.05rem; font-weight: 600; margin: 1.4rem 0 .4rem; color: var(--text); }
    #content h4 { font-size: .95rem; font-weight: 600; margin: 1.2rem 0 .3rem; color: var(--text-muted); }
    #content p  { margin: .6rem 0; }
    #content a  { color: var(--accent); text-decoration: none; }
    #content a:hover { text-decoration: underline; }
    #content ul, #content ol { margin: .5rem 0 .5rem 1.5rem; }
    #content li { margin: .2rem 0; }
    #content hr { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
    #content blockquote {
      border-left: 3px solid var(--accent);
      padding: .4rem .8rem;
      margin: .8rem 0;
      color: var(--text-muted);
      font-style: italic;
      background: var(--accent-bg);
      border-radius: 0 6px 6px 0;
    }
    #content code {
      font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
      font-size: .82rem;
      background: var(--code-bg);
      border: 1px solid var(--border);
      padding: .1rem .35rem;
      border-radius: 4px;
    }
    #content pre {
      background: var(--code-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      overflow-x: auto;
      margin: .75rem 0;
    }
    #content pre code {
      background: none;
      border: none;
      padding: 0;
      font-size: .8rem;
      line-height: 1.6;
    }

    /* ── Tables ───────────────────────────────────── */
    #content table {
      width: 100%;
      border-collapse: collapse;
      margin: 1rem 0;
      font-size: .85rem;
      overflow-x: auto;
      display: block;
    }
    #content th {
      background: var(--accent-bg);
      color: var(--accent);
      font-weight: 600;
      padding: .5rem .75rem;
      text-align: left;
      border: 1px solid var(--border);
      white-space: nowrap;
    }
    #content td {
      padding: .45rem .75rem;
      border: 1px solid var(--border);
      vertical-align: top;
    }
    #content tr:nth-child(even) td { background: rgba(0,0,0,.02); }
    @media (prefers-color-scheme: dark) {
      #content tr:nth-child(even) td { background: rgba(255,255,255,.03); }
    }

    /* ── Figures / images ─────────────────────────── */
    #content img {
      max-width: 100%;
      border-radius: 8px;
      border: 1px solid var(--border);
      margin: .75rem 0;
      cursor: zoom-in;
      display: block;
      box-shadow: var(--shadow);
      transition: opacity .15s;
    }
    #content img:hover { opacity: .88; }

    /* ── Lightbox ─────────────────────────────────── */
    #lightbox {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,.85);
      z-index: 200;
      align-items: center;
      justify-content: center;
      cursor: zoom-out;
    }
    #lightbox.open { display: flex; }
    #lightbox img {
      max-width: 92vw;
      max-height: 92vh;
      object-fit: contain;
      border-radius: 8px;
      box-shadow: 0 4px 40px rgba(0,0,0,.6);
    }
    #lightbox-close {
      position: absolute;
      top: 1rem; right: 1rem;
      background: rgba(255,255,255,.15);
      border: none;
      color: #fff;
      font-size: 1.5rem;
      width: 40px; height: 40px;
      border-radius: 50%;
      cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      transition: background .15s;
    }
    #lightbox-close:hover { background: rgba(255,255,255,.3); }

    /* ── Cheat sheet embed ────────────────────────── */
    .cheatsheet-embed {
      background: var(--surface);
      border-radius: 10px;
      padding: 1.5rem;
      border: 1px solid var(--border);
    }
    .cheatsheet-embed h1 { border-bottom: 2px solid var(--accent); }

    /* ── Responsive ───────────────────────────────── */
    @media (max-width: 768px) {
      #menu-btn { display: flex; align-items: center; }
      #sidebar {
        position: fixed;
        left: -100%;
        top: var(--header-h);
        height: calc(100vh - var(--header-h));
        z-index: 95;
        transition: left .2s ease;
        width: 82vw;
        max-width: 300px;
      }
      #sidebar.open { left: 0; }
      #sidebar-overlay.open { display: block; }
      #main { padding: 1.25rem 1rem; }
    }
  </style>
</head>
<body>
<div id="app">

  <!-- Header -->
  <header id="header">
    <button id="menu-btn" aria-label="Toggle menu">&#9776;</button>
    <h1>AAU CE2 — AI Study Hub</h1>
    <span class="badge">ML · DNN · RL</span>
  </header>

  <div id="body">
    <!-- Sidebar overlay (mobile) -->
    <div id="sidebar-overlay"></div>

    <!-- Sidebar -->
    <nav id="sidebar">
${sidebarHtml}
    </nav>

    <!-- Main content -->
    <main id="main">
      <div id="content"></div>
    </main>
  </div>
</div>

<!-- Lightbox -->
<div id="lightbox" role="dialog" aria-modal="true" aria-label="Figure lightbox">
  <button id="lightbox-close" aria-label="Close">&times;</button>
  <img id="lightbox-img" src="" alt="" />
</div>

<script>
  // ── Page data (pre-rendered HTML) ──────────────────────────────────────────
  const PAGES = ${pagesJson};

  // ── DOM refs ────────────────────────────────────────────────────────────────
  const contentEl   = document.getElementById('content');
  const lightbox    = document.getElementById('lightbox');
  const lightboxImg = document.getElementById('lightbox-img');
  const sidebar     = document.getElementById('sidebar');
  const overlay     = document.getElementById('sidebar-overlay');
  const menuBtn     = document.getElementById('menu-btn');

  // ── Navigation ──────────────────────────────────────────────────────────────
  function getPageId() {
    const hash = location.hash.replace('#', '').trim();
    return PAGES[hash] ? hash : '${firstPageId}';
  }

  function navigate(id) {
    history.pushState(null, '', '#' + id);
    renderPage(id);
  }

  function renderPage(id) {
    const html = PAGES[id];
    if (!html) return;
    contentEl.innerHTML = html;
    // Scroll to top
    document.getElementById('main').scrollTop = 0;
    // Update active link
    document.querySelectorAll('.nav-link').forEach(a => {
      a.classList.toggle('active', a.dataset.page === id);
    });
    // Attach lightbox to images
    contentEl.querySelectorAll('img').forEach(img => {
      img.addEventListener('click', () => openLightbox(img.src, img.alt));
    });
    // Close mobile sidebar
    closeSidebar();
  }

  // ── Lightbox ────────────────────────────────────────────────────────────────
  function openLightbox(src, alt) {
    lightboxImg.src = src;
    lightboxImg.alt = alt || '';
    lightbox.classList.add('open');
    document.body.style.overflow = 'hidden';
  }
  function closeLightbox() {
    lightbox.classList.remove('open');
    document.body.style.overflow = '';
    lightboxImg.src = '';
  }

  lightbox.addEventListener('click', e => { if (e.target === lightbox) closeLightbox(); });
  document.getElementById('lightbox-close').addEventListener('click', closeLightbox);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

  // ── Mobile sidebar ──────────────────────────────────────────────────────────
  function openSidebar()  { sidebar.classList.add('open'); overlay.classList.add('open'); }
  function closeSidebar() { sidebar.classList.remove('open'); overlay.classList.remove('open'); }
  menuBtn.addEventListener('click', () => sidebar.classList.contains('open') ? closeSidebar() : openSidebar());
  overlay.addEventListener('click', closeSidebar);

  // ── Link interception ───────────────────────────────────────────────────────
  document.getElementById('sidebar').addEventListener('click', e => {
    const link = e.target.closest('.nav-link');
    if (!link) return;
    e.preventDefault();
    navigate(link.dataset.page);
  });

  // ── Hash routing ────────────────────────────────────────────────────────────
  window.addEventListener('popstate', () => renderPage(getPageId()));
  window.addEventListener('hashchange', () => renderPage(getPageId()));

  // ── Init ────────────────────────────────────────────────────────────────────
  renderPage(getPageId());
</script>
</body>
</html>
`;

fs.writeFileSync(path.join(DIST, 'index.html'), html, 'utf8');
console.log(`  Written dist/index.html (${Math.round(html.length / 1024)} KB)`);

console.log('Build complete! Output in webapp/dist/');
console.log('  Serve with: node webapp/server.js');
console.log('  Or:         npx serve webapp/dist -l 6767');
