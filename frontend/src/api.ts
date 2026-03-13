const BASE = import.meta.env.VITE_API_BASE ?? ''

export function openChatStream(
  q: string,
  sessionId: string,
  userProfile: string,
  onEvent: (ev: Record<string, unknown>) => void,
  onDone: () => void,
  onError: (msg: string) => void,
): () => void {
  const p = new URLSearchParams({ q, session_id: sessionId, user_profile: userProfile })
  const es = new EventSource(`${BASE}/api/chat/stream?${p}`)
  es.onmessage = (e) => {
    if (e.data === '[DONE]') { es.close(); onDone(); return }
    try { onEvent(JSON.parse(e.data)) } catch { /* ignore parse errors */ }
  }
  es.onerror = () => { es.close(); onError('连接中断，请重试') }
  return () => es.close()
}

export async function clearSession(sessionId: string): Promise<void> {
  await fetch(`${BASE}/api/session/clear`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  })
}

export async function fetchStatus() {
  return fetch(`${BASE}/api/status`).then(r => r.json())
}

export async function fetchMemory(sessionId: string) {
  return fetch(`${BASE}/api/session/memory?session_id=${encodeURIComponent(sessionId)}`).then(r => r.json())
}
