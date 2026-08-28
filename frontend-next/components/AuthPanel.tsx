"use client";
import React, { useState } from "react";
import { api, ApiError, saveIdentity } from "@/lib/api";
import type { Identity } from "@/lib/types";
import { Button, Card } from "./ui";

export function AuthPanel({ onAuthed }: { onAuthed: (id: Identity) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("admin@acme.com");
  const [password, setPassword] = useState("demo1234");
  const [tenant, setTenant] = useState("Acme Inc");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit() {
    setBusy(true);
    setError("");
    try {
      const id =
        mode === "register"
          ? await api.register(email, password, tenant)
          : await api.login(email, password);
      saveIdentity(id);
      onAuthed(id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto mt-24 max-w-md">
      <div className="mb-6 text-center">
        <div className="text-3xl font-bold">🧠 InsightRAG</div>
        <p className="mt-2 text-sm text-muted">
          Agentic Multimodal RAG — smart LLM / RAG / Web routing, hybrid retrieval,
          and a performance deep-dive.
        </p>
      </div>
      <Card>
        <div className="mb-4 flex gap-2">
          {(["login", "register"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`flex-1 rounded-lg px-3 py-2 text-sm capitalize ${
                mode === m ? "bg-accent text-white" : "bg-panel2 text-muted"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
        <div className="space-y-3">
          <Field label="Email" value={email} onChange={setEmail} />
          <Field label="Password" type="password" value={password} onChange={setPassword} />
          {mode === "register" && (
            <Field label="Organization" value={tenant} onChange={setTenant} />
          )}
          {error && <div className="text-sm text-bad">{error}</div>}
          <Button onClick={submit} disabled={busy} className="w-full">
            {busy ? "…" : mode === "register" ? "Create account" : "Sign in"}
          </Button>
        </div>
      </Card>
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  type = "text",
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs uppercase tracking-wide text-muted">{label}</span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-border bg-panel2 px-3 py-2 text-sm outline-none focus:border-accent"
      />
    </label>
  );
}
