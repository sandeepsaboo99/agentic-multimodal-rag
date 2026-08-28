"use client";
import React, { useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { EvalReport } from "@/lib/types";
import { Button, Card, Metric } from "./ui";

const DEFAULT_DATASET = JSON.stringify(
  [
    { question: "What is the primary goal described in the document?", expected_source: "", expected_answer: "" },
    { question: "What are the total figures shown in the financial table?", expected_source: "", expected_answer: "" },
  ],
  null,
  2,
);

export function EvalTab({ token }: { token: string }) {
  const [judge, setJudge] = useState<"proxy" | "llm">("proxy");
  const [text, setText] = useState(DEFAULT_DATASET);
  const [report, setReport] = useState<EvalReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function run() {
    setBusy(true);
    setErr("");
    setReport(null);
    let items: Record<string, string>[];
    try {
      items = JSON.parse(text);
    } catch (e) {
      setErr(`Invalid JSON: ${String(e)}`);
      setBusy(false);
      return;
    }
    try {
      setReport(await api.runEval(token, items, judge));
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
          Run a repeatable eval set and measure retrieval recall, groundedness, correctness,
          latency, and cost. Choose a judge backend:
        </p>
        <div className="mb-3 flex gap-2">
          {(["proxy", "llm"] as const).map((j) => (
            <button
              key={j}
              onClick={() => setJudge(j)}
              className={`rounded-lg px-3 py-1.5 text-sm ${
                judge === j ? "bg-accent text-white" : "bg-panel2 text-muted"
              }`}
              title={
                j === "proxy"
                  ? "Deterministic lexical metrics (free, CI-stable)"
                  : "LLM-as-judge via Groq (needs GROQ_API_KEY; higher quality)"
              }
            >
              {j === "proxy" ? "Proxy (lexical)" : "LLM-as-judge"}
            </button>
          ))}
        </div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={8}
          spellCheck={false}
          className="w-full rounded-lg border border-border bg-panel2 p-3 font-mono text-xs outline-none focus:border-accent"
        />
        <div className="mt-3">
          <Button onClick={run} disabled={busy}>
            {busy ? `Running (${judge})…` : "Run evaluation"}
          </Button>
        </div>
        {err && <div className="mt-2 text-sm text-bad">{err}</div>}
      </Card>

      {report && (
        <>
          <div className="text-sm text-muted">
            Judge used: <span className="font-semibold text-slate-100">{report.judge}</span>
            {report.judge !== judge && " (fell back — no Groq key)"}
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Metric label="Retrieval recall" value={report.retrieval_recall} />
            <Metric label="Groundedness" value={report.groundedness} />
            <Metric label="Answer correctness" value={report.answer_correctness} />
            <Metric label="Citation correctness" value={report.citation_correctness} />
          </div>
          <div className="grid grid-cols-3 gap-3">
            <Metric label="Avg latency" value={`${report.avg_latency_ms} ms`} />
            <Metric label="Avg cost" value={`$${report.avg_cost_usd.toFixed(5)}`} />
            <Metric label="Failure rate" value={report.failure_rate} />
          </div>
          <Card>
            <div className="mb-2 text-sm font-semibold">Per-item details</div>
            <div className="space-y-2">
              {report.details.map((d, i) => (
                <details key={i} className="rounded-lg border border-border bg-panel2 p-2 text-xs">
                  <summary className="cursor-pointer">
                    {String(d.question ?? d.error ?? "item")}{" "}
                    {d.route ? <span className="text-muted">· {String(d.route)}</span> : null}
                  </summary>
                  <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-muted">
                    {JSON.stringify(d, null, 2)}
                  </pre>
                </details>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
