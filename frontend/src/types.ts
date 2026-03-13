export type ToolName = 'rag_search' | 'grep_search' | 'web_search'

export interface ToolCall {
  name: ToolName | string
  query?: string
  keywords?: string
  car_model?: string
}

export interface ToolResultMeta {
  car_model?: string
  query?: string
  keywords?: string
  result_count?: number
  total_before_filter?: number
  score_threshold?: number
  scores?: number[]
  truncated?: boolean
}

export interface ToolResultItem {
  id: string
  name: string
  query: string
  result: {
    content: string
    success: boolean
    metadata: ToolResultMeta
    latency_ms: number
  }
}

export type StreamEvent =
  | { type: 'rewriting' }
  | { type: 'clarify'; message: string }
  | { type: 'refined'; query: string }
  | { type: 'tool_calling'; calls: ToolCall[] }
  | { type: 'tool_done'; results: ToolResultItem[] }
  | { type: 'done'; answer: string; tool_results: ToolResultItem[] | null }
  | { type: 'error'; message: string }

export type TracedEvent = StreamEvent & { ts: number }

export interface Trace {
  original_query: string
  refined_query: string
  elapsed: number
  events: TracedEvent[]
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  tool_calls?: ToolCall[]
  trace?: Trace
  streaming?: boolean
}

export interface SystemStatus {
  rag: { models: Record<string, number | string> }
  api_keys: Record<string, boolean>
  models?: { executor?: string; qwen?: string }
}

export interface MemoryState {
  facts: string[]
  global_user_info: {
    budget: string
    family: string
    preferences: string
    focus_models: string[]
    raw: string
  }
  recent_messages: Array<{ role: string; content: string }>
}
