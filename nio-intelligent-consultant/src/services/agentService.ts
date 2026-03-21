import { Message, ToolCall, Trace, TracedEvent, TraceSummary } from "../types";

const SESSION_ID_KEY = "nio_session_id";

export function getSessionId(): string {
  let id = localStorage.getItem(SESSION_ID_KEY);
  if (!id) {
    id = Math.random().toString(36).substring(2, 11);
    localStorage.setItem(SESSION_ID_KEY, id);
  }
  return id;
}

export function setSessionId(id: string): void {
  localStorage.setItem(SESSION_ID_KEY, id);
}

export function newSessionId(): string {
  const id = Math.random().toString(36).substring(2, 11);
  localStorage.setItem(SESSION_ID_KEY, id);
  return id;
}

export const processChat = async (
  messages: Message[],
  onUpdate: (updatedMessages: Message[]) => void,
  options?: { userProfile?: string }
): Promise<void> => {
  const lastMessage = messages[messages.length - 1];
  if (lastMessage.role !== "user") return;

  const sessionId = getSessionId();
  const assistantMessageId = Math.random().toString(36).substring(7);
  const t0 = Date.now();

  const assistantMsg: Message = {
    id: assistantMessageId,
    role: "assistant",
    content: "",
    timestamp: Date.now(),
    toolCalls: [],
  };

  let currentMessages: Message[] = [...messages, assistantMsg];
  onUpdate(currentMessages);

  const events: TracedEvent[] = [];
  let refinedQuery = lastMessage.content;

  const updateAssistant = (updater: (msg: Message) => Message) => {
    currentMessages = currentMessages.map((m) =>
      m.id === assistantMessageId ? updater({ ...m }) : m
    );
    onUpdate(currentMessages);
  };

  const params = new URLSearchParams({
    q: lastMessage.content,
    session_id: sessionId,
    user_profile: options?.userProfile ?? "",
  });
  const url = `/api/chat/stream?${params}`;

  return new Promise<void>((resolve) => {
    const es = new EventSource(url);

    es.onmessage = (e: MessageEvent) => {
      if (e.data === "[DONE]") {
        es.close();
        resolve();
        return;
      }

      let ev: { type: string; [key: string]: any };
      try {
        ev = JSON.parse(e.data);
      } catch {
        return;
      }

      const ts = (Date.now() - t0) / 1000;
      events.push({ ...ev, ts });

      switch (ev.type) {
        case "rewriting":
          updateAssistant((msg) => ({
            ...msg,
            toolCalls: [{ name: "query_rewriter", args: {}, status: "pending" }],
          }));
          break;

        case "refined":
          refinedQuery = ev.query ?? refinedQuery;
          updateAssistant((msg) => ({
            ...msg,
            toolCalls: (msg.toolCalls || []).map((t) =>
              t.name === "query_rewriter"
                ? { ...t, status: "success", result: ev.query }
                : t
            ),
          }));
          break;

        case "clarify":
          updateAssistant((msg) => ({
            ...msg,
            content: ev.message,
            toolCalls: [],
          }));
          break;

        case "tool_calling": {
          const newCalls: ToolCall[] = (ev.calls || []).map((c: any) => ({
            name: c.name,
            args: c,
            status: "pending" as const,
          }));
          updateAssistant((msg) => ({
            ...msg,
            toolCalls: [...(msg.toolCalls || []), ...newCalls],
          }));
          break;
        }

        case "tool_done": {
          const results: any[] = ev.results || [];
          updateAssistant((msg) => {
            const updatedToolCalls = [...(msg.toolCalls || [])];
            results.forEach((r: any) => {
              const toolName = r.name;
              let idx = -1;
              for (let i = updatedToolCalls.length - 1; i >= 0; i--) {
                if (
                  updatedToolCalls[i].name === toolName &&
                  updatedToolCalls[i].status === "pending"
                ) {
                  idx = i;
                  break;
                }
              }
              if (idx >= 0) {
                const resultContent =
                  r.result?.content ?? JSON.stringify(r.result);
                updatedToolCalls[idx] = {
                  ...updatedToolCalls[idx],
                  status: "success",
                  result: resultContent,
                };
              }
            });
            return { ...msg, toolCalls: updatedToolCalls };
          });
          break;
        }

        case "done": {
          const trace: Trace = {
            original_query: lastMessage.content,
            refined_query: refinedQuery,
            elapsed: (Date.now() - t0) / 1000,
            events: [...events],
            trace_summary: ev.trace_summary as TraceSummary | undefined,
          };
          updateAssistant((msg) => ({
            ...msg,
            content: ev.answer || "",
            trace,
          }));
          break;
        }

        case "error":
          updateAssistant((msg) => ({
            ...msg,
            content:
              ev.message ||
              "I'm sorry, I encountered an error. Please try again.",
          }));
          es.close();
          resolve();
          break;
      }
    };

    es.onerror = () => {
      updateAssistant((msg) => ({
        ...msg,
        content:
          msg.content ||
          "I'm sorry, I encountered an error. Please try again.",
      }));
      es.close();
      resolve();
    };
  });
};
