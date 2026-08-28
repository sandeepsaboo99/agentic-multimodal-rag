// Shared types mirroring the FastAPI backend contracts (app/models/schemas.py).

export interface Identity {
  access_token: string;
  token_type: string;
  tenant_id: string;
  user_id: string;
  workspace_id: string;
  role: string;
}

export interface DocumentOut {
  id: string;
  filename: string;
  status: string;
  version: number;
  n_chunks: number;
  n_tables: number;
  n_images: number;
  error?: string;
}

export interface Citation {
  document_id: string;
  filename: string;
  page: number | null;
  modality: string;
  score: number;
  snippet: string;
}

export interface TraceStep {
  name: string;
  latency_ms: number;
  rss_delta_mb: number;
  heap_peak_mb: number;
  meta: Record<string, unknown>;
}

export interface Trace {
  request_id: string;
  route: string;
  model: string;
  total_tokens: number;
  total_latency_ms: number;
  cost_usd?: number;
  steps: TraceStep[];
}

export interface ChatResponse {
  request_id: string;
  route: string;
  route_reason: string;
  answer: string;
  citations: Citation[];
  trace: Trace;
}

export interface AnalyticsSummary {
  requests: number;
  route_mix: Record<string, number>;
  latency_ms: { p50: number; p95: number; avg: number; max: number };
  tokens_total: number;
  cost_usd_total: number;
  feedback: { up: number; down: number; score: number | null };
  stage_latency_avg_ms: Record<string, number>;
  stage_heap_peak_avg_mb: Record<string, number>;
  process_memory_mb: number;
  cpu_percent: number;
}

export interface HealthInfo {
  embedding_model: string;
  embedding_fallback: boolean;
  reranker_fallback: boolean;
  llm_available: boolean;
  llm_model: string;
}

export interface TraceRow {
  request_id: string;
  route: string;
  route_reason: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_latency_ms: number;
  cost_usd: number;
  created_at: string;
  steps: TraceStep[];
  query: string;
}

export interface EvalReport {
  n: number;
  judge: string;
  retrieval_recall: number;
  groundedness: number;
  answer_correctness: number;
  citation_correctness: number;
  avg_latency_ms: number;
  avg_cost_usd: number;
  failure_rate: number;
  details: Record<string, unknown>[];
}
