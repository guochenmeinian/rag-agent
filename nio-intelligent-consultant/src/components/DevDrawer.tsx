import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, ChevronDown, ChevronRight } from 'lucide-react';
import { Message, SystemStatus, MemoryState, TracedEvent, ToolResultItem, TraceSummary } from '../types';
import { fetchStatus, fetchMemory, clearSession } from '../lib/api';

type DevTab = 'trace' | 'memory' | 'system';

// ── Helpers ────────────────────────────────────────────────────────────────

function ScoreChip({ score }: { score: number }) {
  const color =
    score >= 0.6 ? 'bg-emerald-100 text-emerald-700' :
    score >= 0.4 ? 'bg-amber-100 text-amber-700' :
    'bg-red-100 text-red-700';
  return (
    <span className={`inline-block px-2 py-0.5 rounded-md text-[11px] font-bold ${color}`}>
      {score.toFixed(3)}
    </span>
  );
}

function ToolResultBox({ r }: { r: ToolResultItem }) {
  const [open, setOpen] = useState(false);
  const meta = r.result?.metadata ?? {};
  const scores: number[] = meta.scores ?? [];
  const n = meta.result_count ?? 0;
  const total = meta.total_before_filter ?? n;

  return (
    <div className="ml-4 mt-2 border border-slate-100 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-4 py-2.5 bg-slate-50 hover:bg-slate-100 transition-colors text-left gap-3"
      >
        <div className="min-w-0">
          <span className="text-xs font-semibold text-slate-600 block truncate">{r.name}</span>
          {r.query && <span className="text-[11px] text-slate-400 block truncate">{r.query.slice(0, 40)}</span>}
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {n > 0 && <span className="text-[11px] text-slate-400">{n}/{total} 片段</span>}
          {open ? <ChevronDown size={12} className="text-slate-400" /> : <ChevronRight size={12} className="text-slate-400" />}
        </div>
      </button>
      {scores.length > 0 && (
        <div className="px-4 py-2 flex flex-wrap gap-1.5 border-t border-slate-100 bg-white">
          {scores.map((s, j) => <ScoreChip key={j} score={s} />)}
        </div>
      )}
      {open && (
        <div className="px-4 py-3 text-xs text-slate-500 whitespace-pre-wrap break-words border-t border-slate-100 max-h-48 overflow-y-auto leading-relaxed">
          {r.result?.content ?? ''}
        </div>
      )}
    </div>
  );
}

function TraceEventRow({ ev }: { ev: TracedEvent }) {
  const ts = ev.ts.toFixed(2);

  const dot = (color: string) => (
    <div className={`w-2 h-2 rounded-full flex-shrink-0 mt-0.5 ${color}`} />
  );

  if (ev.type === 'rewriting') return (
    <div className="flex items-start gap-3 py-2">
      {dot('bg-slate-300')}
      <div className="flex-1">
        <span className="text-sm text-slate-600 font-medium">查询改写</span>
      </div>
      <span className="text-xs text-slate-400 flex-shrink-0">{ts}s</span>
    </div>
  );

  if (ev.type === 'refined') return (
    <div className="pl-5 pb-2">
      <span className="text-xs text-blue-500 italic">→ {ev.query}</span>
    </div>
  );

  if (ev.type === 'clarify') return (
    <div className="flex items-start gap-3 py-2">
      {dot('bg-blue-400')}
      <div className="flex-1">
        <span className="text-sm text-slate-600 font-medium">澄清</span>
        <p className="text-xs text-slate-400 mt-0.5">{ev.message.slice(0, 60)}{ev.message.length > 60 ? '…' : ''}</p>
      </div>
      <span className="text-xs text-slate-400 flex-shrink-0">{ts}s</span>
    </div>
  );

  if (ev.type === 'tool_calling') {
    const calls = ev.calls ?? [];
    const batchSize = ev.batch_size ?? calls.length;
    const iterLabel = ev.iteration !== undefined ? ` · 第${ev.iteration + 1}轮` : '';
    return (
      <div className="py-2">
        <div className="flex items-start gap-3">
          {dot('bg-violet-400')}
          <div className="flex-1">
            <span className="text-sm text-slate-600 font-medium">
              工具调用 · {batchSize} 个{batchSize > 1 ? '并行' : ''}{iterLabel}
            </span>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {calls.map((c: any, i: number) => {
                if (c.name === 'rag_search')
                  return <span key={i} className="text-xs px-2.5 py-1 rounded-lg bg-blue-50 text-blue-600 font-medium border border-blue-100">📚 {c.car_model} · {(c.query ?? '').slice(0, 14)}</span>;
                if (c.name === 'grep_search')
                  return <span key={i} className="text-xs px-2.5 py-1 rounded-lg bg-violet-50 text-violet-600 font-medium border border-violet-100">🔍 {c.car_model} · {(c.keywords ?? '').slice(0, 14)}</span>;
                return <span key={i} className="text-xs px-2.5 py-1 rounded-lg bg-emerald-50 text-emerald-600 font-medium border border-emerald-100">🌐 {(c.query ?? '').slice(0, 20)}</span>;
              })}
            </div>
          </div>
          <span className="text-xs text-slate-400 flex-shrink-0">{ts}s</span>
        </div>
      </div>
    );
  }

  if (ev.type === 'tool_done') {
    const results: ToolResultItem[] = ev.results ?? [];
    const s = ev.summary;
    const errLabel = s && s.error_count > 0 ? ` · ${s.error_count} 个失败` : '';
    return (
      <div className="py-2">
        <div className="flex items-start gap-3">
          {dot('bg-violet-400')}
          <div className="flex-1">
            <span className="text-sm text-slate-600 font-medium">
              返回结果 · {results.length} 条
              {errLabel && <span className="text-red-500 ml-1">⚠️{errLabel}</span>}
            </span>
            {results.map((r, i) => <ToolResultBox key={i} r={r as ToolResultItem} />)}
          </div>
          <span className="text-xs text-slate-400 flex-shrink-0">{ts}s</span>
        </div>
      </div>
    );
  }

  if (ev.type === 'done') return (
    <div className="flex items-start gap-3 py-2">
      {dot('bg-emerald-500')}
      <span className="text-sm text-slate-600 font-medium">回答完成</span>
      <span className="text-xs text-slate-400 ml-auto flex-shrink-0">{ts}s</span>
    </div>
  );

  if (ev.type === 'error') return (
    <div className="flex items-start gap-3 py-2">
      {dot('bg-red-400')}
      <span className="text-sm text-red-500 font-medium">错误</span>
      <span className="text-xs text-slate-400 ml-auto flex-shrink-0">{ts}s</span>
    </div>
  );

  return null;
}

function TraceSummaryBar({ s }: { s: TraceSummary }) {
  const flags = [
    s.grep_rag_fallback_used && '检索回退',
    s.force_direct_used && '强制回答',
  ].filter(Boolean).join(' · ');

  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 px-1 py-2 mb-2 text-[11px] text-slate-400 border-b border-slate-50">
      <span>🔁 {s.tool_call_batches} 批 · {s.tool_call_count} 次调用</span>
      {s.tool_error_count > 0 && (
        <span className="text-red-400">⚠️ {s.tool_error_count} 个工具失败</span>
      )}
      <span>🪙 {s.usage.total_tokens} tokens</span>
      {flags && <span className="text-amber-500">{flags}</span>}
    </div>
  );
}

function TraceTurn({ trace, turnNum }: { trace: any; turnNum: number }) {
  const [open, setOpen] = useState(turnNum === 0);
  return (
    <div className="border border-slate-100 rounded-2xl overflow-hidden mb-4">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-5 py-4 bg-slate-50 hover:bg-slate-100 transition-colors text-left"
      >
        <div className="min-w-0 flex-1 mr-4">
          <span className="text-[10px] uppercase tracking-widest font-bold text-slate-400 block mb-0.5">
            Turn {turnNum + 1}
          </span>
          <span className="text-sm font-semibold text-slate-700 block truncate">
            {trace.original_query}
          </span>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <span className="text-xs text-slate-400 font-medium">{trace.elapsed.toFixed(1)}s</span>
          {open ? <ChevronDown size={14} className="text-slate-400" /> : <ChevronRight size={14} className="text-slate-400" />}
        </div>
      </button>
      {open && (
        <div className="px-5 py-3 divide-y divide-slate-50 border-t border-slate-100">
          {trace.trace_summary && <TraceSummaryBar s={trace.trace_summary} />}
          {trace.events.map((ev: TracedEvent, i: number) => (
            <TraceEventRow key={i} ev={ev} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Sub-panels ─────────────────────────────────────────────────────────────

function TracePanel({ messages }: { messages: Message[] }) {
  const turns = messages
    .filter((m) => m.role === 'assistant' && m.trace)
    .map((m) => m.trace!)
    .reverse();

  if (turns.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <div className="w-12 h-12 rounded-2xl bg-slate-100 flex items-center justify-center mb-4">
          <span className="text-xl">📋</span>
        </div>
        <p className="text-sm text-slate-400 font-medium">发送消息后查看执行轨迹</p>
      </div>
    );
  }
  return (
    <div>
      {turns.map((t, i) => (
        <TraceTurn key={i} trace={t} turnNum={turns.length - 1 - i} />
      ))}
    </div>
  );
}

function MemoryPanel({ sessionId }: { sessionId: string }) {
  const [mem, setMem] = useState<MemoryState | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchMemory(sessionId).then(setMem).finally(() => setLoading(false));
  }, [sessionId]);

  if (loading) return (
    <div className="flex items-center justify-center py-16">
      <p className="text-sm text-slate-400">加载中…</p>
    </div>
  );
  if (!mem) return (
    <div className="flex items-center justify-center py-16">
      <p className="text-sm text-slate-400">无法获取记忆状态</p>
    </div>
  );

  const gui = mem.global_user_info;
  const profile =
    [gui.budget && `预算 ${gui.budget}`, gui.family, gui.preferences]
      .filter(Boolean)
      .join('；') || gui.raw;

  return (
    <div className="space-y-6">
      {profile && (
        <section>
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">用户背景</p>
          <p className="text-sm text-slate-600 bg-slate-50 rounded-xl px-4 py-3 leading-relaxed">{profile}</p>
        </section>
      )}

      <section>
        <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
          已记录事实 <span className="text-slate-300 font-normal">({mem.facts.length})</span>
        </p>
        {mem.facts.length > 0 ? (
          <div className="space-y-2">
            {mem.facts.map((f, i) => (
              <div key={i} className="flex gap-3 bg-slate-50 rounded-xl px-4 py-2.5">
                <span className="text-slate-300 flex-shrink-0 mt-0.5">·</span>
                <span className="text-sm text-slate-600">{f}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-400">暂无</p>
        )}
      </section>

      <section>
        <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
          近期消息 <span className="text-slate-300 font-normal">({mem.recent_messages.length})</span>
        </p>
        {mem.recent_messages.length > 0 ? (
          <div className="space-y-3">
            {mem.recent_messages
              .slice()
              .reverse()
              .map((m, i) => (
                <div key={i} className="border border-slate-100 rounded-xl px-4 py-3">
                  <p className="text-[11px] uppercase tracking-wider font-bold text-slate-400 mb-1.5">
                    {m.role === 'user' ? '用户' : '助手'}
                  </p>
                  <p className="text-sm text-slate-600 leading-relaxed line-clamp-3">{m.content}</p>
                </div>
              ))}
          </div>
        ) : (
          <p className="text-sm text-slate-400">暂无</p>
        )}
      </section>
    </div>
  );
}

function SystemPanel({
  sessionId,
  userProfile,
  onProfileChange,
  onClear,
}: {
  sessionId: string;
  userProfile: string;
  onProfileChange: (p: string) => void;
  onClear: () => void;
}) {
  const [status, setStatus] = useState<SystemStatus | null>(null);

  useEffect(() => {
    fetchStatus().then(setStatus).catch(() => {});
  }, []);

  const models = status?.rag?.models ?? {};

  return (
    <div className="space-y-6">
      {/* Session */}
      <section>
        <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">会话</p>
        <div className="bg-slate-50 rounded-xl px-4 py-3 mb-3">
          <p className="text-[11px] text-slate-400 font-bold uppercase tracking-wider mb-1">Session ID</p>
          <p className="text-sm text-slate-600 font-mono">{sessionId}</p>
        </div>
        <textarea
          value={userProfile}
          onChange={(e) => onProfileChange(e.target.value)}
          placeholder="用户偏好（可选）&#10;例：预算50万，家用，注重续航"
          rows={3}
          className="w-full text-sm text-slate-600 bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 resize-none focus:outline-none focus:border-blue-300 placeholder:text-slate-300 leading-relaxed"
        />
        <button
          onClick={onClear}
          className="mt-3 w-full text-sm font-semibold text-red-500 border border-red-200 bg-red-50 hover:bg-red-100 rounded-xl py-2.5 transition-colors"
        >
          清空对话与记忆
        </button>
      </section>

      {/* Knowledge base */}
      <section>
        <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">知识库</p>
        {Object.keys(models).length > 0 ? (
          <div className="flex flex-wrap gap-2">
            {Object.entries(models).map(([m, n]) => (
              <span key={m} className="text-xs px-3 py-1.5 rounded-xl bg-blue-50 text-blue-600 font-semibold border border-blue-100">
                {m} · {n} chunks
              </span>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-400">未加载，检查 data/ 目录</p>
        )}
      </section>

      {/* Models */}
      {status?.models && (
        <section>
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">模型</p>
          <div className="space-y-2">
            {[['推理引擎', status.models.executor], ['改写模型', status.models.qwen]].map(
              ([label, val]) =>
                val && (
                  <div key={label} className="flex items-center justify-between bg-slate-50 rounded-xl px-4 py-3">
                    <span className="text-sm text-slate-500 font-medium">{label}</span>
                    <span className="text-sm text-slate-600 font-mono">{val}</span>
                  </div>
                )
            )}
          </div>
        </section>
      )}

      {/* API Keys */}
      {status?.api_keys && (
        <section>
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">API Keys</p>
          <div className="space-y-2">
            {Object.entries(status.api_keys).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between bg-slate-50 rounded-xl px-4 py-3">
                <div className="flex items-center gap-3">
                  <div className={`w-2 h-2 rounded-full flex-shrink-0 ${v ? 'bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.4)]' : 'bg-slate-300'}`} />
                  <span className="text-sm text-slate-600 font-medium">{k}</span>
                </div>
                <span className={`text-xs font-semibold ${v ? 'text-emerald-600' : 'text-slate-400'}`}>
                  {v ? '已配置' : '未配置'}
                </span>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

// ── Main DevDrawer ─────────────────────────────────────────────────────────

interface DevDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  messages: Message[];
  sessionId: string;
  userProfile: string;
  onProfileChange: (p: string) => void;
  onClear: () => void;
}

export const DevDrawer: React.FC<DevDrawerProps> = ({
  isOpen,
  onClose,
  messages,
  sessionId,
  userProfile,
  onProfileChange,
  onClear,
}) => {
  const [activeTab, setActiveTab] = useState<DevTab>('trace');

  const tabs: [DevTab, string][] = [
    ['trace', '执行轨迹'],
    ['memory', '记忆'],
    ['system', '系统'],
  ];

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 z-30 bg-slate-900/20 backdrop-blur-[2px]"
          />
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            className="fixed right-0 top-0 h-full w-[580px] bg-white z-40 shadow-2xl flex flex-col"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-5 border-b border-slate-100">
              <div>
                <span className="text-xs text-slate-400 font-semibold uppercase tracking-wider block mb-0.5">
                  Developer
                </span>
                <span className="text-base font-black text-slate-800">调试面板</span>
              </div>
              <button
                onClick={onClose}
                className="p-2.5 hover:bg-slate-100 rounded-xl transition-colors text-slate-400 hover:text-slate-700"
              >
                <X size={18} />
              </button>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-slate-100 px-5 pt-2 gap-1">
              {tabs.map(([t, label]) => (
                <button
                  key={t}
                  onClick={() => setActiveTab(t)}
                  className={`relative px-4 py-2.5 text-sm font-semibold transition-colors rounded-t-lg ${
                    activeTab === t ? 'text-slate-800' : 'text-slate-400 hover:text-slate-600'
                  }`}
                >
                  {label}
                  {activeTab === t && (
                    <motion.div
                      layoutId="dev-tab-indicator"
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-slate-800 rounded-full"
                    />
                  )}
                </button>
              ))}
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6">
              {activeTab === 'trace' && <TracePanel messages={messages} />}
              {activeTab === 'memory' && (
                <MemoryPanel sessionId={sessionId} key={messages.length} />
              )}
              {activeTab === 'system' && (
                <SystemPanel
                  sessionId={sessionId}
                  userProfile={userProfile}
                  onProfileChange={onProfileChange}
                  onClear={onClear}
                />
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};
