"use client";
import React, { useCallback, useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError } from "@/lib/api";
import type { AnalyticsSummary, HealthInfo, TraceRow } from "@/lib/types";
import { Button, CHART_COLORS, Card, Metric } from "./ui";

const tooltipStyle = { background: "#121826", border: "1px solid #243049", fontSize: 12 };

export function AnalyticsTab({ token }: { token: string }) {
  const [s, setS] = useState<AnalyticsSummary | null>(null);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [traces, setTraces] = useState<TraceRow[]>([]);
  const [err, setErr] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [sum, h, tr] = await Promise.all([
        api.analyticsSummary(token),
        api.analyticsHealth(token),
        api.analyticsTraces(token, 50),
      ]);
      setS(sum);
      setHealth(h);
      setTraces(tr);
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    }
  }, [token]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  if (err) return <div className="text-sm text-bad">{err}</div>;
  if (!s || !health) return <div className="text-sm text-muted">Loading…</div>;

  const routeData = Object.entries(s.route_mix).map(([name, value]) => ({ name, value }));
  const stageLat = Object.entries(s.stage_latency_avg_ms)
    .map(([stage, ms]) => ({ stage, ms }))
    .sort((a, b) => a.ms - b.ms);
  const stageMem = Object.entries(s.stage_heap_peak_avg_mb)
    .map(([stage, mb]) => ({ stage, mb }))
    .sort((a, b) => a.mb - b.mb);
  const latencyTrend = [...traces].reverse().map((t, i) => ({ i, ms: t.total_latency_ms }));

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">📊 Performance Deep-Dive</h2>
        <Button variant="ghost" onClick={refresh}>
          🔄 Refresh
        </Button>
      </div>

      {/* subsystem health — explains WHY the numbers look the way they do */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric label="LLM" value={health.llm_available ? "live" : "offline-stub"} sub={health.llm_model} />
        <Metric label="Embeddings" value={health.embedding_fallback ? "fallback" : "model"} sub={health.embedding_model} />
        <Metric label="Reranker" value={health.reranker_fallback ? "fallback" : "cross-encoder"} />
        <Metric label="Process RSS" value={`${s.process_memory_mb} MB`} sub={`cpu ${s.cpu_percent}%`} />
      </div>

      {s.requests === 0 ? (
        <Card>
          <div className="text-sm text-muted">
            No requests yet — ask something in the Chat tab to populate analytics.
          </div>
        </Card>
      ) : (
        <>
          {/* KPI row */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
            <Metric label="Requests" value={s.requests} />
            <Metric label="p50 latency" value={`${s.latency_ms.p50} ms`} />
            <Metric label="p95 latency" value={`${s.latency_ms.p95} ms`} />
            <Metric label="Tokens" value={s.tokens_total.toLocaleString()} />
            <Metric label="Cost" value={`$${s.cost_usd_total.toFixed(4)}`} />
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <div className="mb-2 text-sm font-semibold">
                Route distribution <span className="text-muted">(RAG vs WEB vs LLM)</span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={routeData} dataKey="value" nameKey="name" innerRadius={55} outerRadius={90} label>
                      {routeData.map((_, i) => (
                        <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={tooltipStyle} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card>
              <div className="mb-2 text-sm font-semibold">
                Avg latency by pipeline stage <span className="text-muted">(find the bottleneck)</span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stageLat} layout="vertical" margin={{ left: 30 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#243049" />
                    <XAxis type="number" tick={{ fontSize: 10, fill: "#8b97ad" }} />
                    <YAxis type="category" dataKey="stage" width={120} tick={{ fontSize: 10, fill: "#8b97ad" }} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="ms" fill="#37d399" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card>
              <div className="mb-2 text-sm font-semibold">
                Peak heap by stage (MB) <span className="text-muted">(memory hotspots)</span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stageMem} layout="vertical" margin={{ left: 30 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#243049" />
                    <XAxis type="number" tick={{ fontSize: 10, fill: "#8b97ad" }} />
                    <YAxis type="category" dataKey="stage" width={120} tick={{ fontSize: 10, fill: "#8b97ad" }} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="mb" fill="#f5b74e" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card>
              <div className="mb-2 text-sm font-semibold">
                Latency trend <span className="text-muted">(oldest → newest)</span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={latencyTrend}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#243049" />
                    <XAxis dataKey="i" tick={{ fontSize: 10, fill: "#8b97ad" }} />
                    <YAxis tick={{ fontSize: 10, fill: "#8b97ad" }} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Line type="monotone" dataKey="ms" stroke="#5b8cff" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>

          <Card>
            <div className="mb-2 text-sm">
              <span className="font-semibold">User feedback:</span> 👍 {s.feedback.up} · 👎{" "}
              {s.feedback.down} · satisfaction{" "}
              {s.feedback.score != null ? `${(s.feedback.score * 100).toFixed(0)}%` : "—"}
            </div>
          </Card>

          <Card>
            <div className="mb-2 text-sm font-semibold">Recent request traces</div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-muted">
                  <tr>
                    <th className="py-1">time</th>
                    <th>route</th>
                    <th>latency ms</th>
                    <th>tokens</th>
                    <th>cost $</th>
                    <th>query</th>
                  </tr>
                </thead>
                <tbody>
                  {traces.map((t) => (
                    <tr key={t.request_id} className="border-t border-border">
                      <td className="py-1 font-mono">{t.created_at.slice(11, 19)}</td>
                      <td>{t.route}</td>
                      <td>{t.total_latency_ms}</td>
                      <td>{t.prompt_tokens + t.completion_tokens}</td>
                      <td>{t.cost_usd}</td>
                      <td className="max-w-xs truncate text-muted">{t.query}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
