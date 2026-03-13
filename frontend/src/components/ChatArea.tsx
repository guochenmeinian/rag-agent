import { useRef, useEffect } from 'react'
import type { ChatMessage } from '../types'
import Message from './Message'

interface Props {
  messages: ChatMessage[]
  streaming: boolean
  statusText: string
  onSend: (text: string) => void
}

export default function ChatArea({ messages, streaming, statusText, onSend }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, statusText])

  function handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  function submit() {
    const text = textareaRef.current?.value.trim() ?? ''
    if (!text || streaming) return
    textareaRef.current!.value = ''
    autoResize()
    onSend(text)
  }

  function autoResize() {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`
  }

  return (
    <div className="chat-column">
      <div className="messages">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="icon">○</div>
            <p>向我询问关于Nio汽车的任何问题</p>
            <span>续航 · 配置 · 对比 · 购车建议</span>
          </div>
        ) : (
          messages.map(m => <Message key={m.id} msg={m} statusText={m.streaming ? statusText : undefined} />)
        )}
        <div ref={bottomRef} />
      </div>
      <div className="input-bar">
        <div className="input-row">
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            placeholder="问我关于Nio汽车的任何问题…"
            rows={1}
            disabled={streaming}
            onKeyDown={handleKey}
            onInput={autoResize}
          />
          <button className="send-btn" disabled={streaming} onClick={submit}>
            {streaming ? '…' : '发送'}
          </button>
        </div>
        <p className="input-hint">Enter 发送 · Shift+Enter 换行</p>
      </div>
    </div>
  )
}
