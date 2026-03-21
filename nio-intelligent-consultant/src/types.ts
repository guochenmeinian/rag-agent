export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  toolCalls?: ToolCall[];
  trace?: Trace;
}

export interface ToolCall {
  name: string;
  args: any;
  status: 'pending' | 'success' | 'error';
  result?: any;
}

export interface Vehicle {
  name: string;
  type: string;
  typeZh?: string;
  category: 'SEDAN' | 'SUV';
  description: string;
  descriptionZh?: string;
  image: string;
  color: string;
  specs: {
    acceleration: string;
    range: string;
    power: string;
    torque: string;
  };
  features: string[];
  featuresZh?: string[];
}

// ── Dev / Trace types ──────────────────────────────────────────────────────

export interface ToolDoneSummary {
  tool_names: string[];
  success_count: number;
  error_count: number;
  error_types: Record<string, number>;
  total_latency_ms: number;
}

export interface TraceSummary {
  iterations: number;
  tool_call_batches: number;
  tool_call_count: number;
  tools_used: string[];
  tool_success_count: number;
  tool_error_count: number;
  tool_error_types: Record<string, number>;
  tool_latency_ms: number;
  grep_rag_fallback_used: boolean;
  force_direct_used: boolean;
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
}

export interface ToolResultMeta {
  car_model?: string;
  query?: string;
  keywords?: string;
  result_count?: number;
  total_before_filter?: number;
  scores?: number[];
  truncated?: boolean;
}

export interface ToolResultItem {
  id: string;
  name: string;
  query?: string;
  result: {
    content: string;
    success: boolean;
    metadata: ToolResultMeta;
    latency_ms: number;
  };
}

export type TracedEvent = {
  type: string;
  ts: number;
  [key: string]: any;
};

export interface Trace {
  original_query: string;
  refined_query: string;
  elapsed: number;
  events: TracedEvent[];
  trace_summary?: TraceSummary;
}

export interface SystemStatus {
  rag: { models: Record<string, number | string> };
  api_keys: Record<string, boolean>;
  models?: { executor?: string; qwen?: string };
}

export interface MemoryState {
  facts: string[];
  global_user_info: {
    budget: string;
    family: string;
    preferences: string;
    focus_models: string[];
    raw: string;
  };
  recent_messages: Array<{ role: string; content: string }>;
}
