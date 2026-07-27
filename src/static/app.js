/* FinanceBench RAG Application JavaScript */

document.addEventListener('DOMContentLoaded', () => {
  fetchSampleQuestions();
  fetchExperiments();
  fetchDocuments();
  fetchAvailableModels();
});

// Mark embedding-model options that don't have a built index yet as
// unavailable, so picking one can't silently trigger a multi-hour
// re-ingestion behind the scenes.
async function fetchAvailableModels() {
  const embSelect = document.getElementById('embModelSelect');
  if (!embSelect) return;

  try {
    const res = await fetch('/api/available-models');
    const data = await res.json();

    data.embedding_models.forEach(m => {
      const opt = Array.from(embSelect.options).find(o => o.value === m.value);
      if (opt && !m.ready) {
        opt.disabled = true;
        opt.innerText += ' — not indexed yet';
      }
    });
  } catch (err) {
    console.error("Error loading available models:", err);
  }
}

// Tab Switcher
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

  const targetTab = document.getElementById(tabId);
  if (targetTab) {
    targetTab.classList.add('active');
  }

  const btnMap = {
    'queryTab': 'navQueryTab',
    'experimentsTab': 'navExperimentsTab',
    'docsTab': 'navDocsTab'
  };

  const activeBtn = document.getElementById(btnMap[tabId]);
  if (activeBtn) {
    activeBtn.classList.add('active');
  }
}

// Fetch Sample Questions
async function fetchSampleQuestions() {
  const container = document.getElementById('samplePillsContainer');
  if (!container) return;

  try {
    const res = await fetch('/api/sample-questions');
    const data = await res.json();
    
    container.innerHTML = '';
    data.samples.forEach(sample => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'pill-btn';
      btn.innerText = sample.question;
      btn.onclick = () => fillQuery(sample.question);
      container.appendChild(btn);
    });
  } catch (err) {
    console.error("Error loading sample questions:", err);
  }
}

function fillQuery(text) {
  const input = document.getElementById('queryInput');
  if (input) {
    input.value = text;
    input.focus();
  }
}

// Handle Query Submission
async function handleQuerySubmit(event) {
  event.preventDefault();

  const queryInput = document.getElementById('queryInput');
  const submitBtn = document.getElementById('submitBtn');
  const outputCard = document.getElementById('outputCard');
  const answerBody = document.getElementById('answerBody');
  const latencyTag = document.getElementById('latencyTag');
  const sourcesContainer = document.getElementById('sourcesContainer');
  const kInput = document.getElementById('kInput');
  const embModelSelect = document.getElementById('embModelSelect');
  const genModelSelect = document.getElementById('genModelSelect');

  const question = queryInput.value.trim();
  if (!question) return;

  // UI Loading State
  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span class="loader-spinner"></span> Generating...`;

  outputCard.classList.add('active');
  answerBody.innerHTML = `<div style="display: flex; align-items: center; gap: 0.75rem; color: var(--cyan);"><span class="loader-spinner"></span> Embedding query & retrieving top-${kInput.value} passages...</div>`;
  sourcesContainer.innerHTML = '';

  try {
    const response = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: question,
        k: parseInt(kInput.value, 10) || 5,
        embedding_model: embModelSelect ? embModelSelect.value : undefined,
        generation_model: genModelSelect ? genModelSelect.value : undefined
      })
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.detail || 'Query failed');
    }

    const data = await response.json();

    // Latency + actual config used (first request after switching models pays
    // the model-load cost, which shows up here too so it isn't mistaken for a hang)
    const genShort = (data.config.generation_model || '').split('/').pop();
    latencyTag.innerText = `Latency: ${data.latency_seconds}s · ${data.config.embedding_model.split('/').pop()} · ${genShort} · k=${data.config.top_k}`;

    // Format Markdown & Interactive Citations
    answerBody.innerHTML = formatMarkdown(data.answer);

    // Render Source Passages
    sourcesContainer.innerHTML = '';
    data.passages.forEach(p => {
      const card = document.createElement('div');
      card.className = 'passage-card';
      card.id = `sourceCard_${p.citation_id}`;

      card.innerHTML = `
        <div class="passage-header">
          <div class="passage-doc">
            <span class="citation-badge">[${p.citation_id}]</span>
            <span>${escapeHtml(p.doc_name)}</span>
            <span style="color: var(--text-muted); font-size: 0.8rem;">(p. ${p.page})</span>
          </div>
          <div class="similarity-bar-container">
            <span>Match: ${p.similarity_pct}%</span>
            <div class="similarity-bar">
              <div class="similarity-fill" style="width: ${Math.min(100, Math.max(0, p.similarity_pct))}%;"></div>
            </div>
          </div>
        </div>
        <div class="passage-text">${escapeHtml(p.text)}</div>
      `;

      sourcesContainer.appendChild(card);
    });

  } catch (err) {
    answerBody.innerHTML = `<div style="color: #EF4444; font-weight: 600;">Error: ${escapeHtml(err.message)}</div>`;
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = `<span>Ask RAG</span> <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>`;
  }
}

// Format Markdown Bold and Citations
function formatMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);
  // Replace **bold** with <strong>bold</strong>
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong style="color: #FFF;">$1</strong>');
  // Replace [1] with interactive badge
  html = html.replace(/\[(\d+)\]/g, (match, p1) => {
    return `<span class="citation-tag" onclick="scrollToSource(${p1})" title="Jump to Source ${p1}">${match}</span>`;
  });
  return html;
}

// Scroll & Highlight Source Passage when citation tag clicked
function scrollToSource(citationId) {
  const targetCard = document.getElementById(`sourceCard_${citationId}`);
  if (targetCard) {
    document.querySelectorAll('.passage-card').forEach(c => c.classList.remove('highlighted'));
    targetCard.classList.add('highlighted');
    targetCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

// Fetch Experiment Matrix
async function fetchExperiments() {
  const tbody = document.getElementById('experimentsTableBody');
  if (!tbody) return;

  try {
    const res = await fetch('/api/experiments');
    const data = await res.json();

    tbody.innerHTML = '';
    data.experiments.forEach(exp => {
      const tr = document.createElement('tr');
      
      const isTopDocHit = parseFloat(exp['doc_hit@k']) >= 0.50;
      const isTopMRR = parseFloat(exp['MRR']) >= 0.30;

      tr.innerHTML = `
        <td><strong style="color: var(--text-primary);">${escapeHtml(exp.run)}</strong></td>
        <td><span class="badge-tag">${exp.chunk_size} / ${exp.chunk_overlap}</span></td>
        <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;">${escapeHtml(exp.embedding)}</td>
        <td><strong>${exp.k}</strong></td>
        <td>${escapeHtml(exp.gen_model.split('/')[1] || exp.gen_model)}</td>
        <td style="${isTopDocHit ? 'color: var(--emerald); font-weight: 800;' : ''}">${exp['doc_hit@k']}</td>
        <td>${exp['evidence_recall@k']}</td>
        <td style="${isTopMRR ? 'color: var(--cyan); font-weight: 800;' : ''}">${exp.MRR}</td>
        <td>${exp.numeric_match}</td>
        <td>${exp.token_f1}</td>
      `;

      tbody.appendChild(tr);
    });

    updateMetricCards(data.experiments);
    renderComparisonChart(data.experiments);

  } catch (err) {
    console.error("Error loading experiments table:", err);
  }
}

// Compute the "best" metric cards live from the actual comparison data —
// these used to be hardcoded in index.html, which meant they silently went
// stale the moment the underlying results changed.
function updateMetricCards(experiments) {
  if (!experiments || experiments.length === 0) return;

  const best = (key) => experiments.reduce(
    (top, exp) => (parseFloat(exp[key]) > parseFloat(top[key]) ? exp : top),
    experiments[0]
  );

  const fill = (valueId, runId, exp, key, fmt) => {
    const valueEl = document.getElementById(valueId);
    const runEl = document.getElementById(runId);
    if (valueEl) valueEl.innerText = fmt(parseFloat(exp[key]));
    if (runEl) runEl.innerText = exp.run;
  };

  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  const dec = (v) => v.toFixed(3);

  fill('metricDocHit', 'metricDocHitRun', best('doc_hit@k'), 'doc_hit@k', pct);
  fill('metricMRR', 'metricMRRRun', best('MRR'), 'MRR', dec);
  fill('metricRecall', 'metricRecallRun', best('evidence_recall@k'), 'evidence_recall@k', pct);
  fill('metricNumeric', 'metricNumericRun', best('numeric_match'), 'numeric_match', pct);
}

let _comparisonChart = null;

function renderComparisonChart(experiments) {
  const canvas = document.getElementById('comparisonChart');
  if (!canvas || typeof Chart === 'undefined' || !experiments || experiments.length === 0) return;

  if (_comparisonChart) {
    _comparisonChart.destroy();
  }

  const labels = experiments.map(e => e.run);
  const recall = experiments.map(e => parseFloat(e['evidence_recall@k']));
  const f1 = experiments.map(e => parseFloat(e['token_f1']));

  _comparisonChart = new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Evidence Recall@k',
          data: recall,
          backgroundColor: 'rgba(16, 185, 129, 0.65)',
          borderRadius: 4,
        },
        {
          label: 'Token F1',
          data: f1,
          backgroundColor: 'rgba(6, 182, 212, 0.65)',
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#94A3B8', maxRotation: 45, minRotation: 45 }, grid: { display: false } },
        y: { beginAtZero: true, ticks: { color: '#94A3B8' }, grid: { color: 'rgba(255,255,255,0.06)' } },
      },
      plugins: {
        legend: { labels: { color: '#F8FAFC' } },
      },
    },
  });
}

// Fetch Documents Catalog
async function fetchDocuments() {
  const grid = document.getElementById('docsGrid');
  if (!grid) return;

  try {
    const res = await fetch('/api/documents');
    const data = await res.json();

    const headerEl = document.getElementById('docsHeaderCount');
    if (headerEl) headerEl.innerText = `Indexed SEC Filing Knowledge Base (${data.total_pdfs} PDFs)`;

    const noteEl = document.getElementById('docsShowingNote');
    if (noteEl) {
      noteEl.innerText = data.documents.length < data.total_pdfs
        ? `Showing the first ${data.documents.length} of ${data.total_pdfs} — full list available via /api/documents.`
        : '';
    }

    grid.innerHTML = '';
    data.documents.forEach(doc => {
      const card = document.createElement('div');
      card.style.cssText = `
        background: rgba(13, 19, 34, 0.7);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
      `;

      card.innerHTML = `
        <div style="width: 36px; height: 36px; background: rgba(239, 68, 68, 0.15); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #EF4444; font-weight: 800; font-size: 0.75rem;">
          PDF
        </div>
        <div style="flex: 1; overflow: hidden;">
          <div style="font-weight: 600; font-size: 0.85rem; color: var(--text-primary); text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">${escapeHtml(doc.name)}</div>
          <div style="font-size: 0.75rem; color: var(--text-muted);">${doc.doc_type} • ${doc.size_mb} MB</div>
        </div>
      `;

      grid.appendChild(card);
    });

  } catch (err) {
    console.error("Error loading documents catalog:", err);
  }
}

// Helper: Escape HTML
function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
