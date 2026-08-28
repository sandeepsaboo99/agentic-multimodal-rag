"use client";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { DocumentOut } from "@/lib/types";
import { Button, Card, StatusPill } from "./ui";

const IN_PROGRESS = new Set(["uploaded", "parsing", "chunking", "indexing"]);

export function DocumentsTab({ token }: { token: string }) {
  const [docs, setDocs] = useState<DocumentOut[]>([]);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setDocs(await api.listDocuments(token));
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    }
  }, [token]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // auto-poll while any document is still processing (async ingestion states)
  useEffect(() => {
    if (!docs.some((d) => IN_PROGRESS.has(d.status))) return;
    const id = setInterval(refresh, 2000);
    return () => clearInterval(id);
  }, [docs, refresh]);

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setBusy(true);
    setErr("");
    try {
      for (const f of Array.from(files)) await api.upload(token, f);
      await refresh();
    } catch (e) {
      setErr(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <p className="mb-3 text-sm text-muted">
          Upload returns immediately; parsing/embedding runs asynchronously. Watch each doc move
          through <span className="font-mono">uploaded → parsing → chunking → indexing → ready</span>.
        </p>
        <div className="flex items-center gap-3">
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf"
            multiple
            onChange={(e) => upload(e.target.files)}
            className="text-sm text-muted file:mr-3 file:rounded-lg file:border-0 file:bg-accent file:px-4 file:py-2 file:text-sm file:text-white"
          />
          <Button variant="ghost" onClick={refresh} disabled={busy}>
            🔄 Refresh
          </Button>
        </div>
        {err && <div className="mt-2 text-sm text-bad">{err}</div>}
      </Card>

      <Card>
        {docs.length === 0 ? (
          <div className="text-sm text-muted">No documents yet. Upload a PDF to enable RAG.</div>
        ) : (
          <div className="divide-y divide-border">
            {docs.map((d) => (
              <div key={d.id} className="flex items-center gap-3 py-3">
                <div className="flex-1">
                  <div className="text-sm font-medium">
                    {d.filename} <span className="text-xs text-muted">v{d.version}</span>
                  </div>
                  <div className="text-xs text-muted">
                    {d.n_chunks} text · {d.n_tables} tables · {d.n_images} images
                  </div>
                  {d.error && <div className="text-xs text-bad">{d.error}</div>}
                </div>
                <StatusPill status={d.status} />
                <button
                  onClick={async () => {
                    await api.deleteDocument(token, d.id);
                    refresh();
                  }}
                  className="text-muted hover:text-bad"
                  title="Delete"
                >
                  🗑
                </button>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
