"use client";
import React, { useEffect, useState } from "react";
import { clearIdentity, loadIdentity } from "@/lib/api";
import type { Identity } from "@/lib/types";
import { AuthPanel } from "@/components/AuthPanel";
import { ChatTab } from "@/components/ChatTab";
import { DocumentsTab } from "@/components/DocumentsTab";
import { AnalyticsTab } from "@/components/AnalyticsTab";
import { EvalTab } from "@/components/EvalTab";

type TabKey = "chat" | "documents" | "analytics" | "eval";

const TABS: { key: TabKey; label: string }[] = [
  { key: "chat", label: "💬 Chat" },
  { key: "documents", label: "📄 Documents" },
  { key: "analytics", label: "📊 Analytics" },
  { key: "eval", label: "🧪 Evaluation" },
];

export default function Home() {
  const [identity, setIdentity] = useState<Identity | null>(null);
  const [tab, setTab] = useState<TabKey>("chat");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setIdentity(loadIdentity());
    setReady(true);
  }, []);

  if (!ready) return null;
  if (!identity) return <AuthPanel onAuthed={setIdentity} />;

  const token = identity.access_token;

  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <header className="mb-6 flex items-center justify-between">
        <div>
          <div className="text-xl font-bold">🧠 InsightRAG</div>
          <div className="text-xs text-muted">
            Agentic Multimodal RAG · tenant{" "}
            <span className="font-mono">{identity.tenant_id.slice(0, 8)}…</span> · role{" "}
            {identity.role}
          </div>
        </div>
        <button
          onClick={() => {
            clearIdentity();
            setIdentity(null);
          }}
          className="rounded-lg border border-border bg-panel2 px-3 py-1.5 text-sm hover:bg-border/50"
        >
          Sign out
        </button>
      </header>

      <nav className="mb-6 flex gap-1 rounded-xl border border-border bg-panel p-1">
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`flex-1 rounded-lg px-3 py-2 text-sm font-medium transition ${
              tab === t.key ? "bg-accent text-white" : "text-muted hover:bg-panel2"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {tab === "chat" && <ChatTab token={token} />}
        {tab === "documents" && <DocumentsTab token={token} />}
        {tab === "analytics" && <AnalyticsTab token={token} />}
        {tab === "eval" && <EvalTab token={token} />}
      </main>
    </div>
  );
}
