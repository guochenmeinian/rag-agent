export async function clearSession(sessionId: string): Promise<void> {
  await fetch('/api/session/clear', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export async function fetchStatus() {
  return fetch('/api/status').then((r) => r.json());
}

export async function fetchMemory(sessionId: string) {
  return fetch(`/api/session/memory?session_id=${encodeURIComponent(sessionId)}`).then((r) =>
    r.json()
  );
}

export interface SessionMeta {
  session_id: string;
  preview: string;
  message_count: number;
  last_modified: number;
}

export async function fetchSessions(): Promise<SessionMeta[]> {
  return fetch('/api/sessions').then((r) => r.json());
}

export async function fetchSessionMessages(sessionId: string): Promise<{ role: string; content: string }[]> {
  return fetch(`/api/session/messages?session_id=${encodeURIComponent(sessionId)}`)
    .then((r) => r.json())
    .then((d) => d.recent_messages ?? []);
}
