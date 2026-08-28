import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "InsightRAG — Agentic Multimodal RAG",
  description:
    "Agentic multimodal RAG with a smart LLM/RAG/Web decision tree, hybrid retrieval, and a performance deep-dive.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
