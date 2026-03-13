import ReactMarkdown from 'react-markdown'
import type { ChatMessage, ToolCall } from '../types'

interface Props {
  msg: ChatMessage
  statusText?: string
}

function ToolPills({ calls }: { calls: ToolCall[] }) {
  if (!calls.length) return null
  return (
    <div className="pills">
      {calls.map((c, i) => {
        if (c.name === 'rag_search') {
          const q = (c.query ?? '').slice(0, 20)
          return <span key={i} className="pill pill-rag">📚 {c.car_model} · {q}</span>
        }
        if (c.name === 'grep_search') {
          const kw = (c.keywords ?? '').slice(0, 20)
          return <span key={i} className="pill pill-grep">🔍 {c.car_model} · {kw}</span>
        }
        const q = (c.query ?? '').slice(0, 28)
        return <span key={i} className="pill pill-web">🌐 {q}</span>
      })}
    </div>
  )
}

export default function Message({ msg, statusText }: Props) {
  if (msg.role === 'user') {
    return (
      <div className="msg msg-user">
        <div className="bubble">{msg.content}</div>
      </div>
    )
  }

  return (
    <div className="msg msg-assistant">
      {statusText && (
        <div className="status">
          <div className="status-dot" />
          <span>{statusText}</span>
        </div>
      )}
      {(msg.tool_calls?.length ?? 0) > 0 && <ToolPills calls={msg.tool_calls!} />}
      {msg.content ? (
        <div className="prose">
          <ReactMarkdown>{msg.content}</ReactMarkdown>
        </div>
      ) : msg.streaming ? (
        <div className="prose"><span className="cursor" /></div>
      ) : null}
    </div>
  )
}
