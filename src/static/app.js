// ── Config ────────────────────────────────────────────────────────────────────
marked.setOptions({ breaks: true, gfm: true });

// Auto-assign a session_id if the user hasn't typed one, so each browser tab
// gets its own isolated conversation (avoids the shared "__anon__" backend bug).
function getOrInitSessionId() {
  const stored = localStorage.getItem('nio_session_id');
  if (stored) return stored;
  const id = crypto.randomUUID();
  localStorage.setItem('nio_session_id', id);
  return id;
}

// ── State ─────────────────────────────────────────────────────────────────────
let isStreaming = false;
let traces = [];   // [{ turn, query, refined, events[], elapsed }]

// ── DOM refs ──────────────────────────────────────────────────────────────────
const messagesEl   = document.getElementById('messages');
const emptyStateEl = document.getElementById('emptyState');
const chatInput    = document.getElementById('chatInput');
const sendBtn      = document.getElementById('sendBtn');
const clearBtn     = document.getElementById('clearBtn');
const devToggle    = document.getElementById('devToggle');
const devPanel     = document.getElementById('devPanel');
const sessionInput = document.getElementById('sessionInput');
const profileInput = document.getElementById('profileInput');

// Populate session input: prefer user-typed value, fall back to auto-generated UUID
if (!sessionInput.value.trim()) {
  sessionInput.value = getOrInitSessionId();
}
// Persist any manual edits to localStorage
sessionInput.addEventListener('change', () => {
  const val = sessionInput.value.trim();
  if (val) localStorage.setItem('nio_session_id', val);
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function md(text) {
  try { return marked.parse(text); }
  catch { return esc(text).replace(/\n/g,'<br>'); }
}

function pillsHtml(calls) {
  return calls.map(c => {
    if (c.name === 'rag_search')
      return `<span class="pill pill-rag">📚 ${esc(c.car_model)} · ${esc((c.query||'').slice(0,20))}</span>`;
    return `<span class="pill pill-web">🌐 ${esc((c.query||'').slice(0,28))}</span>`;
  }).join('');
}

// ── Message DOM ───────────────────────────────────────────────────────────────

function appendUserMessage(text) {
  emptyStateEl.style.display = 'none';
  const div = document.createElement('div');
  div.className = 'flex justify-end';
  div.innerHTML = `
    <div class="max-w-lg bg-gray-100 rounded-2xl px-4 py-2.5">
      <p class="text-sm whitespace-pre-wrap text-gray-800">${esc(text)}</p>
    </div>`;
  messagesEl.appendChild(div);
  scrollBottom();
}

function appendAssistantMessage() {
  // Returns { id, statusEl, toolsEl, contentEl }
  emptyStateEl.style.display = 'none';
  const id = `msg-${Date.now()}`;
  const div = document.createElement('div');
  div.id = id;
  div.className = 'flex justify-start';
  div.innerHTML = `
    <div class="max-w-2xl w-full space-y-1">
      <div id="${id}-status" class="text-xs text-gray-400 flex items-center gap-1.5 min-h-[16px]"></div>
      <div id="${id}-tools"  class="min-h-0"></div>
      <div id="${id}-content" class="prose text-sm text-gray-800 leading-relaxed">
        <span class="cursor"></span>
      </div>
    </div>`;
  messagesEl.appendChild(div);
  scrollBottom();
  return {
    id,
    statusEl:  document.getElementById(`${id}-status`),
    toolsEl:   document.getElementById(`${id}-tools`),
    contentEl: document.getElementById(`${id}-content`),
  };
}

function setStatus(els, text) {
  els.statusEl.innerHTML = text
    ? `<span class="inline-block w-1.5 h-1.5 rounded-full bg-gray-300 animate-pulse flex-shrink-0"></span>${esc(text)}`
    : '';
}

function finalize(els, answerMarkdown, allCalls) {
  els.statusEl.innerHTML = '';
  els.toolsEl.innerHTML  = pillsHtml(allCalls);
  els.contentEl.innerHTML = md(answerMarkdown);
  scrollBottom();
}

function scrollBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Streaming ─────────────────────────────────────────────────────────────────

function sendMessage(text) {
  if (!text.trim() || isStreaming) return;
  setStreaming(true);

  appendUserMessage(text);
  const els = appendAssistantMessage();

  const params = new URLSearchParams({
    q:            text,
    session_id:   sessionInput.value.trim(),
    user_profile: profileInput ? profileInput.value.trim() : '',
  });

  const es = new EventSource(`/api/chat/stream?${params}`);
  const turn = { turn: traces.length + 1, query: text, refined: text, events: [], elapsed: 0, t0: Date.now() };
  let toolCalls = [];

  es.onmessage = (e) => {
    if (e.data === '[DONE]') {
      es.close();
      turn.elapsed = (Date.now() - turn.t0) / 1000;
      traces.push(turn);
      renderTrace(turn);
      setStreaming(false);
      return;
    }

    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }
    turn.events.push({ ...ev, ts: (Date.now() - turn.t0) / 1000 });

    switch (ev.type) {
      case 'rewriting':
        setStatus(els, '✏️ 理解问题…');
        break;
      case 'refined':
        turn.refined = ev.query;
        if (ev.query !== text) setStatus(els, `💡 ${ev.query.slice(0, 60)}`);
        break;
      case 'tool_calling': {
        toolCalls.push(...(ev.calls || []));
        const names = (ev.calls || []).map(c =>
          c.name === 'rag_search' ? `${c.car_model} 知识库` : '网络搜索'
        ).join(', ');
        setStatus(els, `🔧 查询 ${names}…`);
        els.toolsEl.innerHTML = pillsHtml(toolCalls);
        break;
      }
      case 'tool_done':
        setStatus(els, `✅ 获取 ${(ev.results||[]).length} 条结果，生成回答…`);
        break;
      case 'reflecting':
        setStatus(els, '🔎 验证答案质量…');
        break;
      case 'retry':
        setStatus(els, '↩️ 重新生成…');
        break;
      case 'done':
        finalize(els, ev.answer || '', toolCalls);
        break;
      case 'error':
        finalize(els, `⚠️ 错误：${ev.message}`, []);
        break;
    }
  };

  es.onerror = () => {
    es.close();
    finalize(els, '⚠️ 连接错误，请重试', []);
    setStreaming(false);
  };
}

function setStreaming(val) {
  isStreaming = val;
  sendBtn.disabled   = val;
  chatInput.disabled = val;
  sendBtn.textContent = val ? '…' : '发送';
}

// ── Input handlers ────────────────────────────────────────────────────────────

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (text) { chatInput.value = ''; autoResize(); sendMessage(text); }
  }
});

sendBtn.addEventListener('click', () => {
  const text = chatInput.value.trim();
  if (text) { chatInput.value = ''; autoResize(); sendMessage(text); }
});

function autoResize() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
}
chatInput.addEventListener('input', autoResize);

clearBtn.addEventListener('click', async () => {
  if (!confirm('确认清空对话和记忆？')) return;
  await fetch('/api/session/clear', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionInput.value.trim() }),
  });
  // Reset UI
  while (messagesEl.firstChild) messagesEl.removeChild(messagesEl.firstChild);
  messagesEl.appendChild(emptyStateEl);
  emptyStateEl.style.display = '';
  traces = [];
  const traceTab = document.getElementById('tab-trace');
  traceTab.innerHTML = '<p class="text-xs text-gray-400 text-center mt-6">发送消息后查看执行轨迹</p>';
});

// ── Dev panel ─────────────────────────────────────────────────────────────────

devToggle.addEventListener('click', () => {
  const hidden = devPanel.classList.contains('hidden');
  devPanel.classList.toggle('hidden', !hidden);
  devToggle.textContent = hidden ? '开发者 ↙' : '开发者 ↗';
  if (hidden) loadSystemInfo();
});

document.querySelectorAll('.dev-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.dev-tab').forEach(b => {
      b.classList.toggle('bg-gray-100', b === btn);
      b.classList.toggle('text-gray-700', b === btn);
      b.classList.toggle('text-gray-400', b !== btn);
    });
    document.getElementById('tab-trace').classList.toggle('hidden', tab !== 'trace');
    document.getElementById('tab-system').classList.toggle('hidden', tab !== 'system');
    if (tab === 'system') loadSystemInfo();
  });
});

// ── Trace rendering ───────────────────────────────────────────────────────────

function renderTrace(turn) {
  const container = document.getElementById('tab-trace');
  const placeholder = container.querySelector('p');
  if (placeholder) placeholder.remove();

  const wrap = document.createElement('div');
  wrap.className = 'border border-gray-100 rounded-lg overflow-hidden text-xs';
  wrap.innerHTML = `
    <button onclick="this.nextElementSibling.classList.toggle('hidden')"
      class="w-full flex items-center justify-between px-3 py-2 hover:bg-gray-50 text-left">
      <span class="font-medium text-gray-600 truncate max-w-[220px]">
        Turn ${turn.turn} · ${esc(turn.query.slice(0, 35))}${turn.query.length > 35 ? '…' : ''}
      </span>
      <span class="text-gray-400 ml-2 flex-shrink-0">${turn.elapsed.toFixed(1)}s</span>
    </button>
    <div class="border-t border-gray-100 px-3 py-2 space-y-1.5 bg-gray-50">
      ${traceBodyHtml(turn)}
    </div>`;
  container.insertBefore(wrap, container.firstChild);
}

function traceBodyHtml(turn) {
  let html = '';
  if (turn.refined !== turn.query)
    html += `<p class="text-gray-400 italic mb-1">改写 → ${esc(turn.refined.slice(0, 60))}</p>`;

  for (const ev of turn.events) {
    const ts = (ev.ts || 0).toFixed(2);
    if      (ev.type === 'rewriting')    html += tlRow('bg-purple-300', '✏️ 查询改写', ts);
    else if (ev.type === 'tool_calling') {
      html += tlRow('bg-blue-400', `🔧 工具调用 (${(ev.calls||[]).length}个)`, ts);
      (ev.calls||[]).forEach(c => {
        html += `<div class="pl-3">${c.name === 'rag_search'
          ? `<span class="pill pill-rag">📚 ${esc(c.car_model)} · ${esc((c.query||'').slice(0,20))}</span>`
          : `<span class="pill pill-web">🌐 ${esc((c.query||'').slice(0,28))}</span>`}</div>`;
      });
    }
    else if (ev.type === 'tool_done')   html += tlRow('bg-blue-300', `📦 返回 ${(ev.results||[]).length} 条`, ts);
    else if (ev.type === 'reflecting')  html += tlRow('bg-gray-300',  '🔎 质量校验', ts);
    else if (ev.type === 'retry') {
      html += tlRow('bg-red-400', '↩️ 重试', ts);
      if (ev.feedback) html += `<p class="pl-3 text-red-400">${esc(ev.feedback.slice(0,80))}</p>`;
    }
    else if (ev.type === 'done')        html += tlRow('bg-green-400', '✅ 完成', ts);
  }
  return html || '<p class="text-gray-400">无事件</p>';
}

function tlRow(dot, label, ts) {
  return `<div class="flex items-center gap-2 text-gray-600">
    <span class="w-1.5 h-1.5 rounded-full flex-shrink-0 ${dot}"></span>
    <span class="flex-1">${label}</span>
    <span class="text-gray-400 font-mono">${ts}s</span>
  </div>`;
}

// ── System info ───────────────────────────────────────────────────────────────

async function loadSystemInfo() {
  const el = document.getElementById('systemInfo');
  try {
    const data = await fetch('/api/status').then(r => r.json());
    let html = '';

    html += '<p class="text-gray-400 font-semibold uppercase tracking-wide mb-2">知识库</p>';
    const models = data.rag?.models || {};
    if (Object.keys(models).length) {
      html += '<div class="space-y-1">';
      for (const [m, n] of Object.entries(models))
        html += `<div class="flex justify-between border-b border-gray-50 py-1">
          <span class="text-gray-500">${esc(m)}</span>
          <span class="font-mono text-gray-700">${n} chunks</span></div>`;
      html += '</div>';
    } else {
      html += '<p class="text-gray-300">未加载任何知识库</p>';
    }

    html += '<p class="text-gray-400 font-semibold uppercase tracking-wide mt-4 mb-2">API Keys</p>';
    html += '<div class="space-y-1">';
    for (const [k, v] of Object.entries(data.api_keys || {})) {
      const dot = v
        ? '<span class="w-1.5 h-1.5 rounded-full bg-green-400 inline-block mr-1.5"></span>'
        : '<span class="w-1.5 h-1.5 rounded-full bg-gray-200 inline-block mr-1.5"></span>';
      html += `<div class="flex items-center border-b border-gray-50 py-1">${dot}<span class="text-gray-500">${esc(k)}</span></div>`;
    }
    html += '</div>';

    el.innerHTML = html;
  } catch {
    el.innerHTML = '<p class="text-red-400">加载失败</p>';
  }
}
