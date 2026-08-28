// Typed API client — the SINGLE seam between the React UI and the FastAPI
// backend. Mirrors frontend/api_client.py. All network access lives here so the
// components stay pure view code (same boundary that let us swap Streamlit for
// this app without touching the backend).

import type {
  AnalyticsSummary,
  ChatResponse,
  DocumentOut,
  EvalReport,
  HealthInfo,
  Identity,
  TraceRow,
} from "./types";

const BACKEND =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const TOKEN_KEY = "insightrag_identity";

export function saveIdentity(id: Identity) {
  if (typeof window !== "undefined") localStorage.setItem(TOKEN_KEY, JSON.stringify(id));
}
export function loadIdentity(): Identity | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(TOKEN_KEY);
  return raw ? (JSON.parse(raw) as Identity) : null;
}
export function clearIdentity() {
  if (typeof window !== "undefined") localStorage.removeItem(TOKEN_KEY);
}

export class ApiError extends Error {}

async function req<T>(
  path: string,
  opts: RequestInit = {},
  token?: string,
): Promise<T> {
  const headers = new Headers(opts.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (!(opts.body instanceof FormData) && opts.body) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(`${BACKEND}${path}`, { ...opts, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail ?? JSON.stringify(j);
    } catch {
      /* ignore */
    }
    throw new ApiError(`${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export const api = {
  register: (email: string, password: string, tenant_name: string) =>
    req<Identity>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password, tenant_name }),
    }),

  login: (email: string, password: string) =>
    req<Identity>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  listDocuments: (token: string) =>
    req<DocumentOut[]>("/documents", {}, token),

  upload: (token: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<DocumentOut>("/documents/upload", { method: "POST", body: fd }, token);
  },

  deleteDocument: (token: string, id: string) =>
    req<{ deleted: string }>(`/documents/${id}`, { method: "DELETE" }, token),

  chat: (
    token: string,
    query: string,
    history: { role: string; content: string }[],
    force_route: string | null,
  ) =>
    req<ChatResponse>(
      "/chat",
      { method: "POST", body: JSON.stringify({ query, history, force_route }) },
      token,
    ),

  feedback: (token: string, request_id: string, vote: number) =>
    req<{ ok: boolean }>(
      "/feedback",
      { method: "POST", body: JSON.stringify({ request_id, vote, note: "" }) },
      token,
    ),

  analyticsSummary: (token: string) =>
    req<AnalyticsSummary>("/analytics/summary", {}, token),

  analyticsTraces: (token: string, limit = 50) =>
    req<TraceRow[]>(`/analytics/traces?limit=${limit}`, {}, token),

  analyticsHealth: (token: string) =>
    req<HealthInfo>("/analytics/health", {}, token),

  runEval: (
    token: string,
    items: Record<string, string>[],
    judge: "proxy" | "llm",
  ) =>
    req<EvalReport>(
      "/eval/run",
      { method: "POST", body: JSON.stringify({ items, judge }) },
      token,
    ),
};
