import { useState, useEffect } from 'react'
import type {
  ChatMessage,
  TracedEvent,
  Trace,
  ToolCall,
  ToolResultItem,
  ToolBatchSummary,
  TraceSummary,
  SystemStatus,
  MemoryState,
} from '../types'
import { fetchStatus, fetchMemory } from '../api'

interface Props {
  messages: ChatMessage[]
  sessionId: string
}

type DevTab = 'trace' | 'memory' | 'system'

// ── Trace helpers ──────────────────────────────────────────────────────

function ScoreChip({ score }: { score: number }) {
  const color = score >= 0.6 ? '#16a34a' : score >= 0.4 ? '#d97706' : '#dc2626'
  return (
    <span className="score-chip" style={{ background: `${color}18`, color }}>
      {score.toFixed(3)}
    </span>
  )
}

// Each tool result gets its own open state
function ToolResultBox({ r }: { r: ToolResultItem }) {
  const [open, setOpen] = useState(false)
  const meta = r.result?.metadata ?? {}
  const scores: number[] = meta.scores ?? []
  const n = meta.result_count ?? 0
  const total = meta.total_before_filter ?? n
  const label = `${r.name} · ${(r.query || '').slice(0, 30)}${n > 0 ? ` · ${n}/${total} 片段` : ''}`

  return (
    <div className="tool-result-box" style={{ marginLeft: 14 }}>
      <div className="tool-result-header" onClick={() => setOpen(o => !o)}>
        <span>{label}</span>
        <span style={{ fontSize: 10 }}>{open ? '▲' : '▼'}</span>
      </div>
      {scores.length > 0 && (
        <div className="score-bar">{scores.map((s, j) => <ScoreChip key={j} score={s} />)}</div>
      )}
      {open && (
        <div className="tool-result-body">{r.result?.content ?? ''}</div>
      )}
    </div>
  )
}

function MiniStat({ label, value, tone = 'neutral' }: { label: string; value: string; tone?: 'neutral' | 'ok' | 'warn' }) {
  return (
    <div className={`mini-stat mini-stat-${tone}`}>
      <div className="mini-stat-label">{label}</div>
      <div className="mini-stat-value">{value}</div>
    </div>
  )
}

function ToolBatchSummaryRow({ summary }: { summary: ToolBatchSummary }) {
  const errorTypes = Object.entries(summary.error_types)
    .map(([k, v]) => `${k}×${v}`)
    .join(' · ')

  return (
    <div className="trace-summary-inline">
      <MiniStat label="成功" value={String(summary.success_count)} tone="ok" />
      <MiniStat label="失败" value={String(summary.error_count)} tone={summary.error_count > 0 ? 'warn' : 'neutral'} />
      <MiniStat label="耗时" value={`${summary.total_latency_ms}ms`} />
      {errorTypes && <div className="trace-inline-note">错误类型：{errorTypes}</div>}
    </div>
  )
}

function TraceSummaryCard({ summary }: { summary: TraceSummary }) {
  const errorTypes = Object.entries(summary.tool_error_types)
    .map(([k, v]) => `${k}×${v}`)
    .join(' · ')

  return (
    <div className="trace-summary-card">
      <div className="trace-summary-title">本轮摘要</div>
      <div className="trace-summary-grid">
        <MiniStat label="批次" value={String(summary.tool_call_batches)} />
        <MiniStat label="调用数" value={String(summary.tool_call_count)} />
        <MiniStat label="成功" value={String(summary.tool_success_count)} tone="ok" />
        <MiniStat label="失败" value={String(summary.tool_error_count)} tone={summary.tool_error_count > 0 ? 'warn' : 'neutral'} />
        <MiniStat label="工具耗时" value={`${summary.tool_latency_ms}ms`} />
        <MiniStat label="总 Tokens" value={String(summary.usage.total_tokens)} />
      </div>
      <div className="trace-summary-lines">
        <div>工具：{summary.tools_used.length > 0 ? summary.tools_used.join(' · ') : '无'}</div>
        <div>Fallback：{summary.grep_rag_fallback_used ? 'grep -> rag' : '未触发'}</div>
        <div>Force Direct：{summary.force_direct_used ? '已触发' : '未触发'}</div>
        {errorTypes && <div>错误类型：{errorTypes}</div>}
        <div>
          Tokens：P {summary.usage.prompt_tokens} / C {summary.usage.completion_tokens} / T {summary.usage.total_tokens}
        </div>
      </div>
    </div>
  )
}

function TraceEvent({ ev }: { ev: TracedEvent }) {
  const ts = ev.ts.toFixed(2)

  if (ev.type === 'rewriting') {
    return (
      <div className="tl-row">
        <div className="tl-dot tl-dot-gray" />
        <span className="tl-label">✏️ 查询改写</span>
        <span className="tl-time">{ts}s</span>
      </div>
    )
  }

  // Show the rewritten query inline (indented under rewriting)
  if (ev.type === 'refined') {
    return (
      <div style={{ paddingLeft: 14, marginBottom: 2 }}>
        <span style={{ fontSize: 11, color: 'var(--text3)', fontStyle: 'italic' }}>
          → {ev.query}
        </span>
      </div>
    )
  }

  if (ev.type === 'clarify') {
    return (
      <div className="tl-row">
        <div className="tl-dot tl-dot-info" />
        <span className="tl-label" style={{ color: 'var(--text2)' }}>
          💬 澄清/拒绝 — {ev.message.slice(0, 50)}{ev.message.length > 50 ? '…' : ''}
        </span>
        <span className="tl-time">{ts}s</span>
      </div>
    )
  }

  if (ev.type === 'tool_calling') {
    const calls: ToolCall[] = ev.calls
    return (
      <>
        <div className="tl-row">
          <div className="tl-dot tl-dot-tool" />
          <span className="tl-label">
            🔧 工具调用 ({ev.batch_size ?? calls.length}个{calls.length > 1 ? '并行' : ''})
          </span>
          <span className="tl-time">{ts}s</span>
        </div>
        {typeof ev.iteration === 'number' && (
          <div className="trace-inline-note">第 {ev.iteration + 1} 轮 · {ev.tool_names?.join(' · ') || '工具调用'}</div>
        )}
        <div className="tl-pills">
          {calls.map((c, i) => {
            if (c.name === 'rag_search')
              return <span key={i} className="pill pill-rag">📚 {c.car_model} · {(c.query ?? '').slice(0, 18)}</span>
            if (c.name === 'grep_search')
              return <span key={i} className="pill pill-grep">🔍 {c.car_model} · {(c.keywords ?? '').slice(0, 18)}</span>
            return <span key={i} className="pill pill-web">🌐 {(c.query ?? '').slice(0, 26)}</span>
          })}
        </div>
      </>
    )
  }

  if (ev.type === 'tool_done') {
    return (
      <>
        <div className="tl-row">
          <div className="tl-dot tl-dot-tool" />
          <span className="tl-label">📦 返回结果 ({ev.results.length} 条)</span>
          <span className="tl-time">{ts}s</span>
        </div>
        {ev.summary && <ToolBatchSummaryRow summary={ev.summary} />}
        {ev.results.map((r, i) => <ToolResultBox key={i} r={r} />)}
      </>
    )
  }

  if (ev.type === 'done') {
    return (
      <div className="tl-row">
        <div className="tl-dot tl-dot-done" />
        <span className="tl-label">✅ 回答完成</span>
        <span className="tl-time">{ts}s</span>
      </div>
    )
  }

  if (ev.type === 'error') {
    return (
      <div className="tl-row">
        <div className="tl-dot tl-dot-err" />
        <span className="tl-label" style={{ color: '#dc2626' }}>⚠️ 错误</span>
        <span className="tl-time">{ts}s</span>
      </div>
    )
  }

  return null
}

function TraceTurn({ trace, turnNum }: { trace: Trace; turnNum: number }) {
  const [open, setOpen] = useState(turnNum === 0)
  return (
    <div className="trace-turn">
      <div className="trace-turn-header" onClick={() => setOpen(o => !o)}>
        <span className="trace-turn-title">
          Turn {turnNum + 1} · {trace.original_query.slice(0, 35)}{trace.original_query.length > 35 ? '…' : ''}
        </span>
        <span className="trace-turn-time">{trace.elapsed.toFixed(1)}s</span>
      </div>
      {open && (
        <div className="trace-body">
          {trace.summary && <TraceSummaryCard summary={trace.summary} />}
          {trace.events.map((ev, i) => <TraceEvent key={i} ev={ev} />)}
        </div>
      )}
    </div>
  )
}

// ── Sub-panels ──────────────────────────────────────────────

function TracePanel({ messages }: { messages: ChatMessage[] }) {
  const turns = messages
    .filter(m => m.role === 'assistant' && m.trace)
    .map(m => m.trace!)
    .reverse()

  if (turns.length === 0) {
    return <p className="dev-empty">发送消息后查看执行轨迹</p>
  }
  return (
    <>
      {turns.map((t, i) => (
        <TraceTurn key={i} trace={t} turnNum={turns.length - 1 - i} />
      ))}
    </>
  )
}

// Expandable message bubble for memory panel
function ExpandableMessage({ msg }: { msg: { role: string; content: string } }) {
  const [expanded, setExpanded] = useState(false)
  const LIMIT = 120
  const needsExpand = msg.content.length > LIMIT

  return (
    <div className="mem-message">
      <div className="mem-role">{msg.role === 'user' ? '用户' : '助手'}</div>
      <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
        {expanded ? msg.content : msg.content.slice(0, LIMIT)}
        {!expanded && needsExpand && '…'}
      </div>
      {needsExpand && (
        <button
          onClick={() => setExpanded(e => !e)}
          style={{
            marginTop: 4,
            fontSize: 11,
            color: 'var(--text3)',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            padding: 0,
            textDecoration: 'underline',
          }}
        >
          {expanded ? '收起' : '展开全文'}
        </button>
      )}
    </div>
  )
}

function MemoryPanel({ sessionId }: { sessionId: string }) {
  const [mem, setMem] = useState<MemoryState | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetchMemory(sessionId).then(setMem).finally(() => setLoading(false))
  }, [sessionId])

  if (loading) return <p className="dev-empty">加载中…</p>
  if (!mem) return <p className="dev-empty">无法获取记忆状态</p>

  const gui = mem.global_user_info
  const profile = [gui.budget && `预算${gui.budget}`, gui.family, gui.preferences]
    .filter(Boolean).join('；') || gui.raw

  return (
    <>
      {profile && (
        <div className="mem-section">
          <div className="mem-label">用户背景</div>
          <div className="mem-item">{profile}</div>
        </div>
      )}

      <div className="mem-section">
        <div className="mem-label">事实列表 ({mem.facts.length})</div>
        {mem.facts.length > 0
          ? mem.facts.map((f, i) => <div key={i} className="mem-item">· {f}</div>)
          : <div style={{ fontSize: 11, color: 'var(--text3)' }}>暂无</div>}
      </div>

      <div className="mem-section">
        <div className="mem-label">最近消息 ({mem.recent_messages.length})</div>
        {mem.recent_messages.length > 0
          ? mem.recent_messages.slice().reverse().map((m, i) => (
              <ExpandableMessage key={i} msg={m} />
            ))
          : <div style={{ fontSize: 11, color: 'var(--text3)' }}>暂无</div>}
      </div>
    </>
  )
}

function SystemPanel() {
  const [status, setStatus] = useState<SystemStatus | null>(null)

  useEffect(() => {
    fetchStatus().then(setStatus).catch(() => {})
  }, [])

  if (!status) return <p className="dev-empty">加载中…</p>

  const models = status.rag?.models ?? {}
  return (
    <>
      <div className="sys-section">
        <div className="mem-label">知识库</div>
        {Object.keys(models).length > 0
          ? Object.entries(models).map(([m, n]) => (
              <div key={m} className="sys-row">
                <span className="sys-key">{m}</span>
                <span className="sys-val">{n} chunks</span>
              </div>
            ))
          : <div style={{ fontSize: 11, color: 'var(--text3)' }}>未加载</div>}
      </div>

      {status.models && (
        <div className="sys-section">
          <div className="mem-label">模型</div>
          <div className="sys-row">
            <span className="sys-key">推理引擎</span>
            <span className="sys-val">{status.models.executor ?? '—'}</span>
          </div>
          <div className="sys-row">
            <span className="sys-key">改写</span>
            <span className="sys-val">{status.models.qwen ?? '—'}</span>
          </div>
        </div>
      )}

      <div className="sys-section">
        <div className="mem-label">API Keys</div>
        {Object.entries(status.api_keys ?? {}).map(([k, v]) => (
          <div key={k} className="sys-row">
            <span className="sys-key">
              <span className={`dot ${v ? 'dot-ok' : 'dot-no'}`} />{k}
            </span>
            <span className="sys-val">{v ? '已配置' : '未配置'}</span>
          </div>
        ))}
      </div>
    </>
  )
}

// ── Main DevPanel ──────────────────────────────────────────

export default function DevPanel({ messages, sessionId }: Props) {
  const [activeTab, setActiveTab] = useState<DevTab>('trace')

  return (
    <div className="dev-panel">
      <div className="dev-panel-tabs">
        {([['trace', '📋 执行轨迹'], ['memory', '🧠 记忆'], ['system', '⚙️ 系统']] as [DevTab, string][]).map(([t, label]) => (
          <button
            key={t}
            className={`dev-tab-btn ${activeTab === t ? 'active' : ''}`}
            onClick={() => setActiveTab(t)}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="dev-tab-content">
        {activeTab === 'trace'  && <TracePanel messages={messages} />}
        {activeTab === 'memory' && <MemoryPanel sessionId={sessionId} key={messages.length} />}
        {activeTab === 'system' && <SystemPanel />}
      </div>
    </div>
  )
}
