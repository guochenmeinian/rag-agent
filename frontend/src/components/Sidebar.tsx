import { useState, useEffect } from 'react'
import { fetchStatus } from '../api'
import type { SystemStatus } from '../types'

interface Props {
  sessionId: string
  userProfile: string
  onSessionChange: (id: string) => void
  onProfileChange: (p: string) => void
  onClear: () => void
  onClose: () => void
}

export default function Sidebar({ sessionId, userProfile, onSessionChange, onProfileChange, onClear, onClose }: Props) {
  const [status, setStatus] = useState<SystemStatus | null>(null)

  useEffect(() => {
    fetchStatus().then(setStatus).catch(() => {})
  }, [])

  const models = status?.rag?.models ?? {}
  const loadedModels = Object.keys(models)
  const webOk = status?.api_keys?.['Serper'] ?? false

  return (
    <div className="sidebar">
      <div className="sidebar-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="sidebar-title">Nio AI 助手</div>
          <div className="sidebar-sub">RAG + Agent 智能问答</div>
        </div>
        <button className="icon-btn" onClick={onClose} title="收起侧边栏" style={{ marginTop: -2 }}>✕</button>
      </div>

      <div className="sidebar-body">
        {/* Session */}
        <div>
          <div className="sidebar-section-label">会话</div>
          <input
            className="sidebar-input"
            value={sessionId}
            onChange={e => onSessionChange(e.target.value)}
            placeholder="Session ID"
            style={{ marginBottom: 6 }}
          />
          <textarea
            className="sidebar-input sidebar-textarea"
            value={userProfile}
            onChange={e => onProfileChange(e.target.value)}
            placeholder="用户偏好（可选）&#10;例：预算50万，家用，注重续航"
          />
          <button className="sidebar-btn danger" onClick={onClear} style={{ marginTop: 6 }}>
            清空对话
          </button>
        </div>

        {/* Knowledge base */}
        <div>
          <div className="sidebar-section-label">知识库</div>
          {loadedModels.length > 0 ? (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {loadedModels.map(m => (
                <span key={m} className="pill pill-rag">{m}</span>
              ))}
            </div>
          ) : (
            <p style={{ fontSize: 11, color: 'var(--text3)' }}>未加载，检查 data/ 目录</p>
          )}
        </div>

        {/* Model info */}
        {status?.models && (
          <div>
            <div className="sidebar-section-label">模型</div>
            <div style={{ fontSize: 11, color: 'var(--text2)', lineHeight: 1.7 }}>
              <div>推理：{status.models.executor ?? '—'}</div>
              <div>改写：{status.models.qwen ?? '—'}</div>
            </div>
          </div>
        )}
      </div>

      <div className="sidebar-footer">
        <span className={`dot ${webOk ? 'dot-ok' : 'dot-no'}`} />
        网络搜索 {webOk ? '已启用' : '未配置'}
      </div>
    </div>
  )
}
