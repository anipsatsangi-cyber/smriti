/* ──────────────────────────────────────────────────────────────────────
   Smriti landing page — vanilla JS, no framework.
   Modules: particles, typing, scroll-reveal, counters, nav, live demo.
   ──────────────────────────────────────────────────────────────────── */

// ── Neural Network Particles ──
(function initParticles() {
  const canvas = document.getElementById('particles');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let w = canvas.width = window.innerWidth;
  let h = canvas.height = window.innerHeight;
  
  const PARTICLE_COUNT = Math.min(100, Math.floor((w * h) / 12000));
  const CONNECTION_DISTANCE = 150;
  const particles = [];
  
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles.push({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 2 + 1,
      baseAlpha: Math.random() * 0.5 + 0.2,
      pulse: 0
    });
  }
  
  let mouse = { x: -1000, y: -1000, active: false };
  window.addEventListener('mousemove', e => { 
    mouse.x = e.clientX; 
    mouse.y = e.clientY; 
    mouse.active = true;
  });
  window.addEventListener('mouseout', () => { mouse.active = false; });
  window.addEventListener('resize', () => {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  });

  // Pulses traveling along connections
  const pulses = [];

  function tick() {
    ctx.clearRect(0, 0, w, h);
    
    // Update particles
    for (const p of particles) {
      if (mouse.active) {
        const dx = p.x - mouse.x;
        const dy = p.y - mouse.y;
        const d2 = dx*dx + dy*dy;
        // Mild attraction/repulsion complex to feel organic
        if (d2 < 20000) {
          const f = (20000 - d2) / 200000;
          p.vx -= (dx / Math.sqrt(d2 + 1)) * f;
          p.vy -= (dy / Math.sqrt(d2 + 1)) * f;
          p.pulse = Math.max(p.pulse, f * 2); // glow when near mouse
        }
      }
      
      // Friction and speed limits
      p.vx *= 0.98;
      p.vy *= 0.98;
      
      // Ensure minimum drift
      if (Math.abs(p.vx) < 0.1) p.vx += (Math.random() - 0.5) * 0.05;
      if (Math.abs(p.vy) < 0.1) p.vy += (Math.random() - 0.5) * 0.05;

      p.x += p.vx;
      p.y += p.vy;
      
      // Wrap around
      if (p.x < -50) p.x = w + 50;
      if (p.x > w + 50) p.x = -50;
      if (p.y < -50) p.y = h + 50;
      if (p.y > h + 50) p.y = -50;
      
      p.pulse = Math.max(0, p.pulse - 0.02); // decay glow
    }

    // Draw connections (synapses)
    ctx.lineWidth = 1;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        
        if (dist < CONNECTION_DISTANCE) {
          const opacity = 1 - (dist / CONNECTION_DISTANCE);
          // Base color is a deep bio-purple/blue, glowing to bright blue
          const glow = Math.max(particles[i].pulse, particles[j].pulse);
          const r = Math.floor(99 + glow * 100);
          const g = Math.floor(102 + glow * 100);
          const b = Math.floor(241 + glow * 14);
          
          ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${opacity * 0.3 + glow * 0.4})`;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();

          // Randomly spawn a pulse on an active connection
          if (Math.random() < 0.001) {
            pulses.push({
              p1: particles[i],
              p2: particles[j],
              progress: 0,
              speed: Math.random() * 0.02 + 0.01
            });
          }
        }
      }
      
      // Connect to mouse if active
      if (mouse.active) {
        const dx = particles[i].x - mouse.x;
        const dy = particles[i].y - mouse.y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < CONNECTION_DISTANCE * 1.5) {
          const opacity = 1 - (dist / (CONNECTION_DISTANCE * 1.5));
          ctx.strokeStyle = `rgba(165, 180, 252, ${opacity * 0.5})`;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(mouse.x, mouse.y);
          ctx.stroke();
        }
      }
    }

    // Draw pulses
    for (let i = pulses.length - 1; i >= 0; i--) {
      const p = pulses[i];
      p.progress += p.speed;
      if (p.progress >= 1) {
        p.p2.pulse = 1.0; // target lights up
        pulses.splice(i, 1);
        continue;
      }
      
      const x = p.p1.x + (p.p2.x - p.p1.x) * p.progress;
      const y = p.p1.y + (p.p2.y - p.p1.y) * p.progress;
      
      ctx.beginPath();
      ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw nodes
    for (const p of particles) {
      ctx.beginPath();
      const alpha = p.baseAlpha + p.pulse * 0.5;
      ctx.fillStyle = `rgba(165, 180, 252, ${alpha})`;
      ctx.arc(p.x, p.y, p.r + p.pulse * 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
    
    requestAnimationFrame(tick);
  }
  tick();
})();

// ── Typing effect for hero headline ──
(function initTyping() {
  const el = document.getElementById('typed');
  if (!el) return;
  const phrases = [
    'measured in tokens.',
    'that stays in your browser.',
    'for edge and small devices.',
    'built on cognitive science.',
    'with zero embeddings.',
    'composed via algebra.',
    'persistent across sessions.',
    'portable across runtimes.',
  ];
  let pi = 0, ci = 0, deleting = false;
  function step() {
    const phrase = phrases[pi];
    if (!deleting) {
      ci++;
      el.textContent = phrase.slice(0, ci);
      if (ci >= phrase.length) {
        deleting = true;
        setTimeout(step, 1800);
        return;
      }
    } else {
      ci--;
      el.textContent = phrase.slice(0, ci);
      if (ci === 0) {
        deleting = false;
        pi = (pi + 1) % phrases.length;
        setTimeout(step, 320);
        return;
      }
    }
    setTimeout(step, deleting ? 26 : 56);
  }
  setTimeout(step, 1100);
})();

// ── Scroll reveal ──
(function initScrollReveal() {
  const observer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add('revealed');
        observer.unobserve(e.target);
      }
    }
  }, { threshold: 0.12 });
  document.querySelectorAll('[data-reveal]').forEach(el => observer.observe(el));
})();

// ── Metric counters ──
(function initCounters() {
  const easeOut = t => 1 - Math.pow(1 - t, 3);
  function animateNumber(el, target, suffix) {
    const duration = 1200;
    const start = performance.now();
    function step(now) {
      const elapsed = now - start;
      const t = Math.min(1, elapsed / duration);
      const v = target * easeOut(t);
      // Format: integer if target is integer, otherwise 1 decimal
      const display = (Number.isInteger(target)) ? Math.round(v) : v.toFixed(1);
      el.textContent = `${display}${suffix || ''}`;
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  const observer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        const target = parseFloat(e.target.dataset.target);
        const suffix = e.target.dataset.suffix || '';
        animateNumber(e.target, target, suffix);
        observer.unobserve(e.target);
      }
    }
  }, { threshold: 0.55 });
  document.querySelectorAll('.metric .num').forEach(el => observer.observe(el));
})();

// ── Nav scroll behavior ──
(function initNav() {
  const nav = document.getElementById('nav');
  if (!nav) return;
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 24);
  }, { passive: true });
})();

// ── Live in-browser demo, powered by REAL WASM-compiled Smriti ──
//
// The retrieval logic on this page is the same Rust code that ships
// natively. We compile it to WASM (~230KB), load it in the browser, and
// every Remember/Recall click invokes the real engine — no JavaScript
// simulation. This is the marketing-defining moment: nobody else in the
// agent-memory space can do this because everyone else needs an
// embedding model.
import init, { WasmSmriti } from '../pkg/smriti.js';

async function initLiveDemo() {
  const input = document.getElementById('demo-input');
  const rememberBtn = document.getElementById('demo-remember');
  const recallBtn = document.getElementById('demo-recall');
  const clearBtn = document.getElementById('demo-clear');
  const output = document.getElementById('demo-output');
  const stats = document.getElementById('demo-stats');

  // Graph Canvas setup
  const canvas = document.getElementById('demo-graph-canvas');
  if (!input || !output || !canvas) return;

  // ── Boot the WASM engine ──
  // We do this once per page load. The .wasm file is fetched from
  // /pkg/codegraph_memory_bg.wasm. Cloudflare/Vercel will cache it.
  let smriti = null;
  try {
    await init();
    smriti = new WasmSmriti();
  } catch (err) {
    output.innerHTML = `<div class="demo-empty" style="color:var(--red)">
      Failed to load Smriti WASM engine: ${err.message || err}.<br>
      <em>Try a hard refresh (Ctrl+Shift+R) or check the console.</em>
    </div>`;
    return;
  }

  const ctx = canvas.getContext('2d');
  let cw, ch;

  function resizeCanvas() {
    const parent = canvas.parentElement;
    cw = canvas.width = parent.clientWidth;
    ch = canvas.height = parent.clientHeight;
  }
  window.addEventListener('resize', resizeCanvas);
  // Initial delay to let CSS settle
  setTimeout(resizeCanvas, 100);

  // We keep a JS-side mirror of memories purely for the graph viz.
  // The IDs come from WASM (UUID strings) so we can correlate hits.
  const memories = [];          // [{ id (uuid), text, tokens }]
  const idToNodeIdx = new Map(); // uuid → graph node index
  const nodes = [];
  const edges = [];
  let recalls = 0;
  let totalEfficiencySum = 0;

  // Cheap tokenizer used only for the graph-edge weight (visual layer).
  // The actual retrieval scoring is handled by WASM.
  function tokenize(text) {
    return text
      .toLowerCase()
      .split(/[^a-z0-9_]+/)
      .filter(t => t.length >= 4);
  }

  function tagScore(qtokens, mtokens) {
    if (qtokens.length === 0 || mtokens.length === 0) return 0;
    const q = new Set(qtokens);
    const m = new Set(mtokens);
    let inter = 0;
    for (const t of q) if (m.has(t)) inter++;
    return inter / Math.max(q.size, m.size);
  }

  function appendLine(html, cls) {
    if (output.querySelector('.demo-empty')) output.innerHTML = '';
    const div = document.createElement('div');
    div.className = `demo-line ${cls || ''}`;
    div.innerHTML = html;
    output.appendChild(div);
    output.scrollTop = output.scrollHeight;
  }

  function updateStats() {
    const avgEff = recalls === 0 ? 0 : Math.round(totalEfficiencySum / recalls);
    stats.innerHTML = `
      <span><strong>${memories.length}</strong> memories stored</span>
      <span class="dim">·</span>
      <span><strong>${recalls}</strong> recalls run</span>
      <span class="dim">·</span>
      <span><strong>${avgEff}%</strong> avg token efficiency</span>
    `;
  }
  
  // Graph physics and rendering
  function addNode(uuid, text, tokens) {
    const node = {
      uuid,
      x: cw / 2 + (Math.random() - 0.5) * 50,
      y: ch / 2 + (Math.random() - 0.5) * 50,
      vx: 0, vy: 0,
      text,
      tokens,
      activePulse: 0,
    };

    // Create edges to existing nodes based on shared tokens (visual only)
    for (const other of nodes) {
      const score = tagScore(tokens, other.tokens);
      if (score > 0) {
        edges.push({ source: node, target: other, weight: score });
      }
    }
    idToNodeIdx.set(uuid, nodes.length);
    nodes.push(node);

    // Add a little explosive force when a new node drops
    node.activePulse = 1.0;
  }

  function highlightRecallNodes(hits) {
    // Dim all
    for (const n of nodes) n.activePulse = -0.5;
    // Highlight hits — match by UUID returned from WASM
    for (const h of hits) {
      const idx = idToNodeIdx.get(h.id);
      if (idx !== undefined && nodes[idx]) nodes[idx].activePulse = 1.0;
    }
  }

  function tickGraph() {
    if (!cw || !ch) {
      requestAnimationFrame(tickGraph);
      return;
    }
    ctx.clearRect(0, 0, cw, ch);
    
    // Spring physics
    for (const e of edges) {
      const dx = e.target.x - e.source.x;
      const dy = e.target.y - e.source.y;
      const dist = Math.sqrt(dx*dx + dy*dy) + 0.1;
      const force = (dist - 80) * 0.005 * e.weight;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      
      e.source.vx += fx;
      e.source.vy += fy;
      e.target.vx -= fx;
      e.target.vy -= fy;
    }
    
    // Repulsion and center gravity
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      // Center gravity
      n.vx += (cw/2 - n.x) * 0.001;
      n.vy += (ch/2 - n.y) * 0.001;
      
      for (let j = i + 1; j < nodes.length; j++) {
        const n2 = nodes[j];
        const dx = n2.x - n.x;
        const dy = n2.y - n.y;
        const dist = Math.sqrt(dx*dx + dy*dy) + 0.1;
        if (dist < 100) {
          const force = -20 / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          n.vx += fx; n.vy += fy;
          n2.vx -= fx; n2.vy -= fy;
        }
      }
    }
    
    // Draw edges
    ctx.lineWidth = 1;
    for (const e of edges) {
      const activeScore = Math.max(0, Math.max(e.source.activePulse, e.target.activePulse));
      const r = Math.floor(100 + activeScore * 155);
      const b = Math.floor(150 + activeScore * 105);
      ctx.strokeStyle = `rgba(${r}, 100, ${b}, ${0.15 + e.weight * 0.3 + activeScore * 0.5})`;
      ctx.beginPath();
      ctx.moveTo(e.source.x, e.source.y);
      ctx.lineTo(e.target.x, e.target.y);
      ctx.stroke();
    }
    
    // Draw nodes
    for (const n of nodes) {
      n.vx *= 0.85; n.vy *= 0.85;
      n.x += n.vx; n.y += n.vy;
      
      // Keep in bounds
      n.x = Math.max(10, Math.min(cw - 10, n.x));
      n.y = Math.max(10, Math.min(ch - 10, n.y));
      
      // Decay pulse back to 0
      if (n.activePulse > 0) n.activePulse -= 0.02;
      if (n.activePulse < 0) n.activePulse += 0.02;
      
      ctx.beginPath();
      const isLit = n.activePulse > 0.1;
      const isDim = n.activePulse < -0.1;
      
      if (isLit) {
        ctx.fillStyle = `rgba(245, 158, 11, ${0.8 + n.activePulse * 0.2})`; // Saffron hit
        ctx.shadowColor = 'rgba(245, 158, 11, 0.8)';
        ctx.shadowBlur = 15;
      } else if (isDim) {
        ctx.fillStyle = `rgba(100, 100, 120, 0.3)`; // Dim
        ctx.shadowBlur = 0;
      } else {
        ctx.fillStyle = `rgba(129, 140, 248, 0.8)`; // Indigo default
        ctx.shadowBlur = 0;
      }
      
      ctx.arc(n.x, n.y, isLit ? 6 + n.activePulse * 3 : 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0; // reset
    }
    
    requestAnimationFrame(tickGraph);
  }
  tickGraph();

  // ── Soft caps for the public demo ──
  // The engine itself has no per-memory length limit and no max-memory
  // ceiling — those are policies the application sets. For the public
  // landing-page demo, we want to keep a hostile or careless visitor
  // from tanking their *own* browser tab (and screenshot-shaming us
  // with "Smriti is broken"). These caps don't affect the library;
  // they only apply to this in-browser sandbox.
  const MAX_MEMORY_TEXT_CHARS = 2000;   // single-memory length cap
  const MAX_DEMO_MEMORIES     = 200;    // total-memory cap per tab

  // ── Remember handler — calls real WASM ──
  rememberBtn.addEventListener('click', () => {
    const text = input.value.trim();
    if (!text) return;

    // Cap 1: per-memory text length.
    if (text.length > MAX_MEMORY_TEXT_CHARS) {
      appendLine(
        `<span style="color:var(--yellow)">demo limit</span> · single memory capped at ${MAX_MEMORY_TEXT_CHARS} chars in this sandbox · self-host Smriti for unbounded text`,
        'note'
      );
      return;
    }

    // Cap 2: total memories in this tab.
    if (memories.length >= MAX_DEMO_MEMORIES) {
      appendLine(
        `<span style="color:var(--yellow)">demo limit</span> · ${MAX_DEMO_MEMORIES} memories max in this sandbox · use Reset to start over, or self-host for unbounded growth`,
        'note'
      );
      return;
    }

    try {
      // Reset pulses
      for (const n of nodes) n.activePulse = 0;
      // Call WASM. Returns the new memory's UUID as a string.
      const uuid = smriti.remember(text, 'fact', []);
      memories.push({ id: uuid, text });
      addNode(uuid, text, tokenize(text));
      appendLine(
        `<span style="color:var(--green)">remember</span> · stored <code>${escapeHtml(text)}</code>` +
        ` <span style="color:var(--t3)">[${uuid.slice(0, 8)}…]</span>`,
        'ok'
      );
      input.value = '';
      updateStats();
    } catch (err) {
      appendLine(`<span style="color:var(--red)">error</span> · ${escapeHtml(err.message || String(err))}`, 'note');
    }
  });

  // ── Recall handler — calls real WASM ──
  recallBtn.addEventListener('click', () => {
    const text = input.value.trim();
    if (!text) return;
    if (memories.length === 0) {
      appendLine(
        `<span style="color:var(--yellow)">recall</span> · no memories yet — try Remember first`,
        'note'
      );
      return;
    }
    try {
      // Force consolidation so the neocortex sees recently-added memories
      smriti.consolidate();
      const json = smriti.recall(text, 500);
      const result = JSON.parse(json);
      recalls++;
      const used = result.tokens_used;
      const budget = result.tokens_budget;
      const efficiency = Math.round(100 * (1 - used / budget));
      totalEfficiencySum += efficiency;
      appendLine(
        `<span style="color:var(--accent-bright)">recall</span> · "${escapeHtml(text)}" → ` +
        `<strong>${result.hits.length}</strong> hits, <strong>${used}/${budget}</strong> tokens (${efficiency}% saved)`,
        'hit'
      );
      if (result.hits.length === 0) {
        appendLine(
          `  <em style="color:var(--t3)">no matching memories — Smriti's keyword + graph layer didn't find a hit. Add tags or be more specific.</em>`,
          'note'
        );
        for (const n of nodes) n.activePulse = -0.5;
      } else {
        if (result.verdict === 'unsupported_top' || result.verdict === 'low_confidence') {
          appendLine(
            `  <em style="color:var(--yellow)">Warning: Low confidence match (noise). The engine returned these as a fallback because they tied on graph/decay signals.</em>`,
            'note'
          );
        }
        highlightRecallNodes(result.hits);
        for (const h of result.hits) {
          appendLine(
            `  <span style="color:var(--accent-bright)">[${h.score.toFixed(3)}]</span> ${escapeHtml(h.text)}`,
            'hit'
          );
        }
      }
      input.value = '';
      updateStats();
    } catch (err) {
      appendLine(`<span style="color:var(--red)">error</span> · ${escapeHtml(err.message || String(err))}`, 'note');
    }
  });

  // ── Hard Reset ──
  // The "Clear" button used to only wipe the output panel — leaving the
  // underlying WASM store full of whatever the user had typed. That's
  // misleading and a real privacy footgun: someone pastes confidential
  // text, hits Clear, and walks away thinking it's gone. It isn't.
  //
  // The new behavior: wipe the entire engine via WasmSmriti.reset(),
  // wipe the JS-side mirror used by the graph viz, re-seed the demo
  // memories, and clear the output panel. The user gets a fresh slate.
  //
  // We still scope to a single browser tab — Smriti's WASM build is
  // ephemeral by design (Smriti::new_ephemeral), so cross-visitor data
  // bleed is architecturally impossible. Reset is for the user's own
  // peace of mind.
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      // Soft confirm — short enough not to be annoying, blunt enough to
      // make the privacy model legible.
      const ok = confirm(
        'Reset your local Smriti demo?\n\n' +
        'This wipes every memory and edge in this browser tab.\n' +
        'Your data only ever lived in this tab — nobody else can see it,\n' +
        'including the Smriti team. This action is irreversible.'
      );
      if (!ok) return;

      try {
        const dropped = smriti.reset();
        // Wipe the JS-side mirrors used by the graph viz.
        memories.length = 0;
        nodes.length = 0;
        edges.length = 0;
        idToNodeIdx.clear();
        recalls = 0;
        totalEfficiencySum = 0;
        // Re-seed so the demo isn't visually empty.
        for (const m of seedMemories) {
          try {
            const uuid = smriti.remember(m.text, m.kind, m.tags);
            memories.push({ id: uuid, text: m.text });
            addNode(uuid, m.text, tokenize(m.text));
          } catch (_) { /* ignore seed errors on reset */ }
        }
        try { smriti.consolidate(); } catch (_) { /* ignore */ }
        output.innerHTML = '';
        appendLine(
          `<span style="color:var(--green)">●</span> reset · dropped <strong>${dropped}</strong> memories · re-seeded with <strong>${seedMemories.length}</strong> demo facts`,
          'ok'
        );
        input.value = '';
        updateStats();
      } catch (err) {
        appendLine(
          `<span style="color:var(--red)">error</span> · reset failed: ${escapeHtml(err.message || String(err))}`,
          'note'
        );
      }
    });
  }

  // Pre-seed with a few memories so first-time visitors see something useful
  const seedMemories = [
    { text: "The auth module uses JWT RS256 with 1-hour expiry", kind: 'fact', tags: ['auth', 'security'] },
    { text: "Database is Postgres 15 with read replicas", kind: 'fact', tags: ['db', 'infra'] },
    { text: "Bob is the lead engineer on the auth refactor project", kind: 'fact', tags: ['user', 'auth'] },
    { text: "User prefers concise responses without emojis", kind: 'preference', tags: ['user', 'style'] },
    { text: "We chose Rust for performance and safety", kind: 'decision', tags: ['lang', 'rust'] },
    { text: "Sessions expire after 8 hours", kind: 'fact', tags: ['auth', 'session'] },
  ];
  for (const m of seedMemories) {
    try {
      const uuid = smriti.remember(m.text, m.kind, m.tags);
      memories.push({ id: uuid, text: m.text });
      addNode(uuid, m.text, tokenize(m.text));
    } catch (err) {
      console.warn('Seed memory failed:', err);
    }
  }
  // Run an initial consolidation so seeds are in the neocortex
  try { smriti.consolidate(); } catch (_) { /* ignore */ }

  // Wait a beat before relaxing seed nodes
  setTimeout(() => {
    for (const n of nodes) n.activePulse = 0;
  }, 1000);

  // Replace the placeholder in the output panel with a "ready" line
  appendLine(
    `<span style="color:var(--green)">●</span> Smriti WASM engine loaded · ${seedMemories.length} seed memories · type a query and click Recall`,
    'ok'
  );

  updateStats();

  // Allow Enter to recall
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      if (e.shiftKey) {
        rememberBtn.click();
      } else {
        recallBtn.click();
      }
    }
  });

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[c]);
  }
}

// Kick off the demo. If WASM init fails the user sees a clear error in
// the output panel.
initLiveDemo().catch(err => {
  console.error('Live demo failed to initialize:', err);
});
