"use client";
import React, { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError } from "@/lib/api";
import type { ChatResponse } from "@/lib/types";
import { Button, Card, RouteBadge } from "./ui";

const FORCE = ["auto", "RAG", "WEB", "DIRECT_LLM"] as const;

export function ChatTab({ token }: { token: string }) {
  const [query, setQuery] = useState("");
  const [force, setForce] = useState<(typeof FORCE)[number]>("auto");
  const [history, setHistory] = useState<{ role: string; content: string }[]>([]);
  const [resp, setResp] = useState<ChatResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [showTrace, setShowTrace] = useState(false);

  async function send() {
    if (!query.trim()) return;
    setBusy(true);
    setErr("");
    try {
      const r = await api.chat(token, query, history, force === "auto" ? null : force);
      setResp(r);
      setHistory((h) => [
        ...h,
        { role: "user", content: query },
        { role: "assistant", content: r.answer },
      ]);
      setQuery("");
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <p className="mb-3 text-sm text-muted">
          The agent decides — per question — whether to use your documents (RAG),
          live web search (WEB), or answer directly (DIRECT_LLM).
        </p>
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="e.g. What was total revenue in the Q3 table?"
            className="flex-1 rounded-lg border border-border bg-panel2 px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <select
            value={force}
            onChange={(e) => setForce(e.target.value as (typeof FORCE)[number])}
            className="rounded-lg border border-border bg-panel2 px-2 text-sm"
            title="Override the decision tree to compare behaviours"
          >
            {FORCE.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
          <Button onClick={send} disabled={busy}>
            {busy ? "…" : "Send"}
          </Button>
        </div>
        {err && <div className="mt-2 text-sm text-bad">{err}</div>}
      </Card>

      {resp && (
        <>
          <Card>
            <div className="mb-3 flex items-center gap-3">
              <RouteBadge route={resp.route} />
            </div>
            <div className="mb-3 rounded-lg border border-accent/30 bg-accent/10 px-3 py-2 text-sm">
              🧭 <span className="font-semibold">Why:</span> {resp.route_reason}
            </div>
            <div className="whitespace-pre-wrap text-sm leading-relaxed">{resp.answer}</div>

            <div className="mt-4 flex gap-2">
              <Button variant="ghost" onClick={() => api.feedback(token, resp.request_id, 1)}>
                👍 Helpful
              </Button>
              <Button variant="ghost" onClick={() => api.feedback(token, resp.request_id, -1)}>
                👎 Not helpful
              </Button>
            </div>
          </Card>

          {resp.citations.length > 0 && (
            <Card>
              <div className="mb-2 text-sm font-semibold">Sources</div>
              <div className="space-y-2">
                {resp.citations.map((c, i) => (
                  <details key={i} className="rounded-lg border border-border bg-panel2 p-2">
                    <summary className="cursor-pointer text-sm">
                      <span className="font-mono text-accent">[{i + 1}]</span> {c.filename} ·{" "}
                      {c.modality}
                      {c.page ? ` · p.${c.page}` : ""} · score {c.score}
                    </summary>
                    <p className="mt-2 text-xs text-muted">{c.snippet}</p>
                  </details>
                ))}
              </div>
            </Card>
          )}

          <Card>
            <button
              onClick={() => setShowTrace((s) => !s)}
              className="text-sm font-semibold text-accent"
            >
              🔬 {showTrace ? "Hide" : "Show"} technical trace for THIS request
            </button>
            {showTrace && (
              <div className="mt-3 space-y-4">
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  <TraceStat label="Total latency" value={`${resp.trace.total_latency_ms} ms`} />
                  <TraceStat label="Tokens" value={resp.trace.total_tokens} />
                  <TraceStat label="Model" value={resp.trace.model || "—"} />
                  <TraceStat
                    label="Cost"
                    value={resp.trace.cost_usd != null ? `$${resp.trace.cost_usd}` : "—"}
                  />
                </div>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={resp.trace.steps}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#243049" />
                      <XAxis dataKey="name" tick={{ fontSize: 10, fill: "#8b97ad" }} angle={-15} height={50} />
                      <YAxis tick={{ fontSize: 10, fill: "#8b97ad" }} />
                      <Tooltip
                        contentStyle={{ background: "#121826", border: "1px solid #243049" }}
                      />
                      <Bar dataKey="latency_ms" fill="#5b8cff" name="latency (ms)" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <table className="w-full text-left text-xs">
                  <thead className="text-muted">
                    <tr>
                      <th className="py-1">step</th>
                      <th>latency (ms)</th>
                      <th>peak heap (MB)</th>
                      <th>meta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {resp.trace.steps.map((s, i) => (
                      <tr key={i} className="border-t border-border">
                        <td className="py-1 font-mono">{s.name}</td>
                        <td>{s.latency_ms}</td>
                        <td>{s.heap_peak_mb}</td>
                        <td className="max-w-xs truncate font-mono text-muted">
                          {JSON.stringify(s.meta)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}

function TraceStat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-panel2 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-muted">{label}</div>
      <div className="truncate text-sm font-semibold">{value}</div>
    </div>
  );
}
