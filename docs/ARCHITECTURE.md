# Architecture Deep-Dive

Complements the README. Focuses on request/data flow and design trade-offs.

## Request flow: a chat turn

```
Streamlit ──POST /chat (JWT)──► FastAPI ──► agent.answer_query
                                              │
   1. security.current_principal  ← decode JWT → Principal(tenant,user,ws,role)
   2. _has_corpus(principal)      ← does this tenant have ready docs?
   3. router.decide(...)          ← feature extraction + RETRIEVAL PROBE
        └─ retrieval.retrieve():  dense(Qdrant, ACL-filtered)
                                  + sparse(BM25 over ACL corpus)
                                  → RRF → cross-encoder rerank → top-K, conf
   4. dispatch by route:
        RAG        → reuse probe evidence, format numbered context
        WEB        → websearch.web_search → format results
        DIRECT_LLM → no evidence
   5. generation.generate(...)    ← Groq, grounded system prompt, citations
   6. observability.Trace         ← every step timed + memory-sampled
   7. persist TraceRecord         ← powers Analytics tab
   ◄── ChatResponse{answer, route, route_reason, citations, trace}
```

The **retrieval probe in step 3 is reused in step 4** for the RAG path — we never
retrieve twice. The router's confidence signal *is* the reranker's top score.

## Data flow: ingestion

```
POST /documents/upload
  → hash bytes (versioning gate)
  → store_object(PDF)                     [object storage]
  → INSERT Document(status=uploaded)      [metadata DB]
  → INSERT Job(status=queued)             [job queue]
  → 200 OK  (fast; no parsing on the request path)

worker.run_ingestion_job(job)
  → read PDF from object storage
  → parse_pdf(): PyMuPDF text+images, pdfplumber tables
  → caption images via Groq vision → caption text
  → store extracted images to object storage
  → RecursiveCharacterTextSplitter on text; tables/images atomic
  → EmbeddingProvider.embed(all units)
  → VectorStore.delete_document (idempotent) → upsert with ACL payloads
  → Document.status = ready
```

## Key trade-offs

**Native Qdrant sparse (current) vs in-process BM25 (old).** Sparse retrieval now
uses **native Qdrant sparse vectors**: each chunk stores a term-frequency sparse
vector, the sparse index carries `Modifier.IDF`, and a single `query_points` call
fuses dense + sparse with server-side RRF. This replaced an in-process BM25 scan
that was O(corpus) per query and blind to the index. The client-side RRF remains
only as a fallback for backends that can't fuse server-side. The next upgrade on
the same seam is a learned sparse encoder (SPLADE/BM42) in place of the hashing
term-frequency encoder — the storage/query shape is unchanged.

**Embedded Qdrant vs server.** Embedded (on-disk) mode makes the repo run with
zero infra, but payload indexes are no-ops and only one process may open the path.
Production uses server Qdrant (`QDRANT_URL`) where indexes and concurrency work.

**Deterministic eval proxies vs LLM-judge.** Proxies (token-F1, lexical overlap)
are free and CI-stable but coarse. The `EvalReport` interface is shaped so an
LLM-judge or RAGAS backend drops in without changing callers.

**Router: rules + LLM tie-break vs pure LLM.** A pure-LLM router is a latency and
cost tax on every turn and is non-deterministic. The rule tree handles the
obvious cases instantly and deterministically; the LLM votes only on borderline
cases near the confidence threshold. Best of both: fast, cheap, explainable, with
a smart fallback exactly where ambiguity lives.

**Streamlit now, React later.** Because the UI only talks to the API via
`api_client`, replacing Streamlit with a Next.js frontend is a frontend-only
change — the contract in `app/api` and `models/schemas` is the stable boundary.

## Concurrency & singletons

`EmbeddingProvider`, `Reranker`, `VectorStore`, `GenerationService`, and
`AgenticRouter` are lazily-initialized, thread-safe singletons so expensive model
loads and DB clients happen once per process. Under `uvicorn --workers N` each
worker process holds its own set — memory scales with worker count, which the
Analytics tab's per-process RSS makes visible.
```
