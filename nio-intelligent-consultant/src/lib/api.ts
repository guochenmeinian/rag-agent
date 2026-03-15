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
