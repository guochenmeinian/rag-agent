import { useState, useCallback } from 'react'
import type { ChatMessage, ToolCall, Trace, TracedEvent, TraceSummary } from './types'
import { openChatStream, clearSession } from './api'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import DevPanel from './components/DevPanel'

type Tab = 'chat' | 'dev'

function makeId() { return `${Date.now()}-${Math.random().toString(36).slice(2, 6)}` }

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [streaming, setStreaming] = useState(false)
  const [statusText, setStatusText] = useState('')
  const [tab, setTab] = useState<Tab>('chat')
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const [sessionId, setSessionId] = useState(() => {
    const stored = localStorage.getItem('nio_session_id')
    if (stored) return stored
    const id = crypto.randomUUID().slice(0, 8)
    localStorage.setItem('nio_session_id', id)
    return id
  })
  const [userProfile, setUserProfile] = useState('')

  const handleSessionChange = useCallback((id: string) => {
    setSessionId(id)
    localStorage.setItem('nio_session_id', id)
  }, [])

  const handleSend = useCallback((text: string) => {
    if (!text.trim() || streaming) return

    const userMsg: ChatMessage = { id: makeId(), role: 'user', content: text }
    const assistantId = makeId()
    const assistantMsg: ChatMessage = { id: assistantId, role: 'assistant', content: '', streaming: true }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setStreaming(true)
    setStatusText('正在思考…')

    const t0 = Date.now()
    const toolCalls: ToolCall[] = []
    const events: TracedEvent[] = []
    let refinedQuery = text
    let traceSummary: TraceSummary | undefined

    openChatStream(
      text,
      sessionId,
      userProfile,
      (ev) => {
        const ts = (Date.now() - t0) / 1000
        events.push({ ...(ev as unknown as TracedEvent), ts })

        const upd = (patch: Partial<ChatMessage>) =>
          setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, ...patch } : m))

        switch ((ev as Record<string, unknown>).type) {
          case 'rewriting':
            setStatusText('✏️ 理解问题…')
            break
          case 'refined':
            refinedQuery = (ev as Record<string, unknown>).query as string
            setStatusText(`💡 ${refinedQuery.slice(0, 55)}`)
            break
          case 'tool_calling': {
            const calls: ToolCall[] = ((ev as Record<string, unknown>).calls as ToolCall[]) ?? []
            toolCalls.push(...calls)
            const names = calls.map(c => {
              if (c.name === 'rag_search') return `${c.car_model} 知识库`
              if (c.name === 'grep_search') return `${c.car_model} 精确检索`
              return '网络搜索'
            }).join(', ')
            setStatusText(`🔧 查询 ${names}…`)
            upd({ tool_calls: [...toolCalls] })
            break
          }
          case 'tool_done':
            setStatusText(`✅ 获取 ${(((ev as Record<string, unknown>).results as unknown[]) ?? []).length} 条结果，生成回答…`)
            break
          case 'done': {
            const answer = (ev as Record<string, unknown>).answer as string ?? ''
            traceSummary = (ev as Record<string, unknown>).trace_summary as TraceSummary | undefined
            const trace: Trace = {
              original_query: text,
              refined_query: refinedQuery,
              elapsed: (Date.now() - t0) / 1000,
              events: [...events],
              summary: traceSummary,
            }
            upd({ content: answer, tool_calls: [...toolCalls], trace, streaming: false })
            setStatusText('')
            break
          }
          case 'error':
            upd({ content: `⚠️ ${(ev as Record<string, unknown>).message as string}`, streaming: false })
            setStatusText('')
            break
        }
      },
      () => { setStreaming(false); setStatusText('') },
      (err) => {
        setMessages(prev => prev.map(m =>
          m.id === assistantId ? { ...m, content: `⚠️ ${err}`, streaming: false } : m
        ))
        setStreaming(false)
        setStatusText('')
      },
    )
  }, [streaming, sessionId, userProfile])

  const handleClear = useCallback(async () => {
    if (!confirm('确认清空对话和记忆？')) return
    await clearSession(sessionId)
    setMessages([])
    setStatusText('')
  }, [sessionId])

  return (
    <div className="app">
      {sidebarOpen && (
        <Sidebar
          sessionId={sessionId}
          userProfile={userProfile}
          onSessionChange={handleSessionChange}
          onProfileChange={setUserProfile}
          onClear={handleClear}
          onClose={() => setSidebarOpen(false)}
        />
      )}
      <div className="main">
        <header className="header">
          {!sidebarOpen && (
            <button className="icon-btn" onClick={() => setSidebarOpen(true)} title="打开侧边栏">☰</button>
          )}
          <span className="header-title">○ Nio AI 助手</span>
          <div className="header-spacer" />
          <div className="tab-bar">
            <button className={`tab-btn ${tab === 'chat' ? 'active' : ''}`} onClick={() => setTab('chat')}>
              💬 对话
            </button>
            <button className={`tab-btn ${tab === 'dev' ? 'active' : ''}`} onClick={() => setTab('dev')}>
              🔧 开发者
            </button>
          </div>
        </header>
        <div className={`content ${tab === 'dev' ? 'split' : ''}`}>
          <ChatArea
            messages={messages}
            streaming={streaming}
            statusText={statusText}
            onSend={handleSend}
          />
          {tab === 'dev' && <DevPanel messages={messages} sessionId={sessionId} />}
        </div>
      </div>
    </div>
  )
}
