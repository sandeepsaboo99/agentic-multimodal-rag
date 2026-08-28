// Small presentational primitives shared across tabs.
import React from "react";

export function Card({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-xl border border-border bg-panel p-4 ${className}`}>
      {children}
    </div>
  );
}

export function Metric({
  label,
  value,
  sub,
}: {
  label: string;
  value: React.ReactNode;
  sub?: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-panel2 px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
      {sub && <div className="text-xs text-muted">{sub}</div>}
    </div>
  );
}

const ROUTE_STYLE: Record<string, string> = {
  RAG: "bg-good/15 text-good border-good/40",
  WEB: "bg-accent/15 text-accent border-accent/40",
  DIRECT_LLM: "bg-warn/15 text-warn border-warn/40",
};

export function RouteBadge({ route }: { route: string }) {
  const label =
    route === "RAG"
      ? "🟢 RAG · your documents"
      : route === "WEB"
        ? "🌐 WEB · live search"
        : "💬 DIRECT · model knowledge";
  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 text-sm font-medium ${
        ROUTE_STYLE[route] ?? "border-border text-muted"
      }`}
    >
      {label}
    </span>
  );
}

export function StatusPill({ status }: { status: string }) {
  const map: Record<string, string> = {
    ready: "bg-good/15 text-good",
    failed: "bg-bad/15 text-bad",
    uploaded: "bg-panel2 text-muted",
    parsing: "bg-warn/15 text-warn",
    chunking: "bg-warn/15 text-warn",
    indexing: "bg-accent/15 text-accent",
  };
  return (
    <span className={`rounded-md px-2 py-0.5 text-xs font-mono ${map[status] ?? "bg-panel2 text-muted"}`}>
      {status}
    </span>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  disabled,
  className = "",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "danger";
  disabled?: boolean;
  className?: string;
}) {
  const styles = {
    primary: "bg-accent text-white hover:bg-accent/90",
    ghost: "border border-border bg-panel2 hover:bg-border/50",
    danger: "border border-bad/40 text-bad hover:bg-bad/10",
  }[variant];
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg px-4 py-2 text-sm font-medium transition disabled:opacity-40 ${styles} ${className}`}
    >
      {children}
    </button>
  );
}

export const CHART_COLORS = ["#5b8cff", "#37d399", "#f5b74e", "#ff6b6b", "#b78bff", "#4ec7e0"];
