#!/usr/bin/env node
/**
 * Static file server for AAU CE2 AI Study Webapp
 * Serves webapp/dist/ on port 6767 (0.0.0.0 — LAN accessible)
 *
 * Usage:  node webapp/server.js
 *         PORT=8080 node webapp/server.js   (override port)
 */

const http = require('http');
const fs   = require('fs');
const path = require('path');

const PORT = parseInt(process.env.PORT || '6767', 10);
const DIST = path.resolve(__dirname, 'dist');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css':  'text/css',
  '.js':   'application/javascript',
  '.json': 'application/json',
  '.png':  'image/png',
  '.jpg':  'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif':  'image/gif',
  '.svg':  'image/svg+xml',
  '.webp': 'image/webp',
  '.ico':  'image/x-icon',
  '.txt':  'text/plain',
};

const server = http.createServer((req, res) => {
  // Sanitize URL, default to index.html
  let urlPath = req.url.split('?')[0].split('#')[0];
  if (urlPath === '/' || urlPath === '') urlPath = '/index.html';

  const filePath = path.join(DIST, urlPath);

  // Prevent path traversal
  if (!filePath.startsWith(DIST)) {
    res.writeHead(403); res.end('Forbidden'); return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      if (err.code === 'ENOENT') {
        // SPA fallback: serve index.html for unknown paths
        fs.readFile(path.join(DIST, 'index.html'), (e2, d2) => {
          if (e2) { res.writeHead(404); res.end('Not found'); return; }
          res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
          res.end(d2);
        });
      } else {
        res.writeHead(500); res.end('Server error');
      }
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    const ct = MIME[ext] || 'application/octet-stream';
    res.writeHead(200, {
      'Content-Type': ct,
      'Cache-Control': ext === '.html' ? 'no-cache' : 'public, max-age=3600',
    });
    res.end(data);
  });
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`ERROR: port ${PORT} is already in use. Stop the other process or run: PORT=<other> node webapp/server.js`);
    process.exit(1);
  }
  throw err;
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`AAU CE2 AI Study Hub running at:`);
  console.log(`  http://localhost:${PORT}`);
  console.log(`  http://0.0.0.0:${PORT}  (LAN accessible)`);
  console.log(`Press Ctrl+C to stop.`);
});
