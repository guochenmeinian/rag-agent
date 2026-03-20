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

export interface ToolBatchSummary {
  tool_names: string[]
  success_count: number
  error_count: number
  error_types: Record<string, number>
  total_latency_ms: number
}

export interface TraceSummary {
  iterations: number
  tool_call_batches: number
  tool_call_count: number
  tools_used: string[]
  tool_success_count: number
  tool_error_count: number
  tool_error_types: Record<string, number>
  tool_latency_ms: number
  grep_rag_fallback_used: boolean
  force_direct_used: boolean
  usage: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

export type StreamEvent =
  | { type: 'rewriting'; stage?: string }
  | { type: 'clarify'; message: string }
  | { type: 'refined'; query: string; rewriter_skipped?: boolean }
  | { type: 'tool_calling'; calls: ToolCall[]; iteration?: number; batch_size?: number; tool_names?: string[] }
  | { type: 'tool_done'; results: ToolResultItem[]; iteration?: number; summary?: ToolBatchSummary }
  | { type: 'done'; answer: string; tool_results: ToolResultItem[] | null; usage?: TraceSummary['usage']; trace_summary?: TraceSummary }
  | { type: 'error'; message: string }

export type TracedEvent = StreamEvent & { ts: number }

export interface Trace {
  original_query: string
  refined_query: string
  elapsed: number
  events: TracedEvent[]
  summary?: TraceSummary
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
