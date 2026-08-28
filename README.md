# 🧠 InsightRAG — Agentic Multimodal RAG (Production Architecture)

> **From feature demo → secure, scalable, observable, testable, maintainable product.**
>
> A working, runnable implementation of the *Productionizing a Multimodal RAG
> Application* assignment. It couples a **FastAPI** service backend, a **Streamlit**
> web app, **Groq**-served open-weight LLMs, hybrid retrieval over **Qdrant**, and
> an **agentic decision tree** that decides — per question — whether to answer from
> the model directly, from your documents (RAG), or from live web search.

---

## Table of contents
1. [What this is](#1-what-this-is)
2. [The complex real-life scenario](#2-the-complex-real-life-scenario)
3. [Architecture & service separation](#3-architecture--service-separation)
4. [Component responsibility table](#4-component-responsibility-table)
5. [Multi-tenant data model](#5-multi-tenant-data-model)
6. [Async ingestion workflow](#6-async-ingestion-workflow)
7. [The agentic decision tree (LLM / RAG / Web)](#7-the-agentic-decision-tree-llm--rag--web)
8. [Hybrid retrieval pipeline — step-by-step reasoning](#8-hybrid-retrieval-pipeline--step-by-step-reasoning)
9. [Multimodal design (images + tables)](#9-multimodal-design-images--tables)
10. [Security & multi-tenancy](#10-security--multi-tenancy)
11. [Document versioning & incremental ingestion](#11-document-versioning--incremental-ingestion)
12. [Evaluation & observability (the Analytics tab)](#12-evaluation--observability-the-analytics-tab)
13. [Project structure](#13-project-structure)
14. [Quickstart (local)](#14-quickstart-local)
15. [Production deployment](#15-production-deployment)
16. [Phased roadmap → code map](#16-phased-roadmap--code-map)

---

## 1. What this is

InsightRAG is a **service-oriented** RAG platform, not a monolithic Streamlit
script. The UI holds no business logic; it calls a FastAPI gateway that fans out
to independent **ingestion**, **retrieval**, and **generation** services. Every
request is fully **traced** (latency + memory + tokens + cost per pipeline stage),
which powers a dedicated **performance deep-dive tab**.

**Stack (all open where possible):**

| Concern | Choice | Why |
|---|---|---|
| LLM (text + vision) | **Groq** open-weight Llama models | Fast, cheap, open weights; text + vision + a small router model |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, local) | Free, private, CPU-friendly; **graceful hashing fallback** if offline |
| Reranker | `ms-marco-MiniLM-L-6-v2` cross-encoder | Precise re-scoring of fused candidates |
| Vector DB | **Qdrant** (embedded on-disk *or* server) | Payload-filtered ANN; zero-setup locally, managed cluster in prod |
| Sparse retrieval | **Qdrant native sparse vectors** (IDF modifier) | BM25-style lexical scoring *inside* the DB; fused server-side (no client-side scan) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Boundary-aware splitting (the requested text splitter) |
| PDF parsing | PyMuPDF (text/images) + pdfplumber (tables) | High-fidelity multimodal extraction |
| Web tool | DuckDuckGo (`ddgs`) | No API key; live public info |
| Metadata/ACL DB | SQLAlchemy → SQLite (dev) / Postgres (prod) | System of record for identity, jobs, versions, traces |
| API | FastAPI | Typed, async, auto-docs |
| UI | **Streamlit** (admin/demo) **+ Next.js/React** (production) | Two frontends on one API — proving the boundary is swappable |

---

## 2. The complex real-life scenario

**"Meridian" — a multi-tenant enterprise knowledge assistant.**

Analysts at several client organizations upload dense PDFs — annual reports,
research papers, contracts — containing **prose, financial tables, and charts**.
They ask questions that are a mix of:

- **Document-grounded** — *"What was R&D spend in the Q3 table?"* → must hit the
  exact table cell and cite the page.
- **General knowledge** — *"Explain what EBITDA means."* → the model already knows;
  retrieving would waste latency and risk grounding on irrelevant chunks.
- **Real-time** — *"What is the current 10-year treasury yield?"* → not in any PDF;
  needs live web search.

A naive "always-RAG" app fails all three edge cases differently. InsightRAG's
**agent decides the strategy per question**, isolates each tenant's data, and
exposes exactly *why* it chose what it did — plus the latency/memory/cost of
getting there. That transparency is what makes it trustworthy in production.

---

## 3. Architecture & service separation

```
                              USERS
                                │
                 ┌──────────────┴──────────────┐
                 │   Streamlit UI (thin view)   │   ← swappable for React/Next.js
                 └──────────────┬──────────────┘
                                │ HTTPS + JWT
                 ┌──────────────▼──────────────┐
                 │   FastAPI Gateway / Auth      │   app/main.py
                 └───┬───────────┬───────────┬──┘
        Document APIs│   Chat APIs│  Analytics│Eval APIs
                     │            │           │
        ┌────────────▼───┐   ┌────▼─────────────────────────┐
        │ Object Storage │   │      Agent Orchestrator       │  services/agent.py
        │ (PDFs, images) │   │  ┌─────────────────────────┐  │
        └───────┬────────┘   │  │  Agentic Decision Tree  │  │  services/router.py
                │            │  │  DIRECT_LLM / RAG / WEB  │  │
        ┌───────▼────────┐   │  └───────┬────────┬────────┘  │
        │   Job Queue     │   │      RAG│     WEB │           │
        │ (Jobs table /   │   │  ┌──────▼────┐ ┌──▼────────┐  │
        │  Redis/SQS)     │   │  │ Retrieval │ │  Web tool │  │
        └───────┬────────┘   │  │  Service  │ │(DuckDuckGo)│  │
                │            │  └─────┬─────┘ └───────────┘  │
        ┌───────▼────────┐   │        │                      │
        │ Parser Workers │   │  Dense + Sparse → RRF → Rerank│  services/retrieval.py
        │ text/table/img │   │        │                      │
        │  + embeddings  │   │  ┌─────▼──────┐               │
        └───────┬────────┘   │  │ Generation │ (Groq LLM +   │  services/generation.py
                │            │  │  Service   │  citations)   │
        ┌───────▼────────┐   │  └─────┬──────┘               │
        │     Qdrant     │◄──┘        │                      │
        │ (vectors +ACL  │            ▼                      │
        │   payloads)    │      Answer + Citations + Trace   │
        └───────┬────────┘                                   │
                │                                            │
        ┌───────▼────────────────────────────────────────────▼──┐
        │        Metadata / ACL DB (Postgres) + Trace store       │  models/db.py
        │  users · tenants · documents · versions · jobs · traces │
        └─────────────────────────────────────────────────────────┘
```

**What scales independently:** the **worker** fleet (CPU-heavy parsing/embedding)
scales separately from the **API** (I/O-bound), which scales separately from
**Qdrant** (memory-bound ANN). This is the whole point of the service split.

---

## 4. Component responsibility table

| Component | File(s) | Responsibility |
|---|---|---|
| **Frontend** | `frontend/` (Streamlit) · `frontend-next/` (Next.js/React) | Two thin view layers on one API: auth, chat, upload, analytics, eval. No business logic. |
| **API gateway** | `app/main.py`, `app/api/*` | AuthN/Z, request validation, routing to services, CORS, health. |
| **Auth / identity** | `app/core/security.py` | JWT issue/verify → `Principal(tenant,user,workspace,role)`. |
| **Ingestion service** | `app/services/ingestion.py`, `parsing.py` | Object-store the PDF, parse text/table/image, chunk, embed, index. |
| **Workers** | `worker/ingestion_worker.py` | Consume the job queue; run ingestion with retries/backoff. |
| **Object storage** | `ingestion.store_object` | Raw PDFs + extracted images (local dir → S3/MinIO). |
| **Vector store** | `app/services/vectorstore.py`, `sparse.py` | Qdrant single-collection, payload-partitioned, ACL-filtered; native dense+sparse hybrid with server-side RRF. |
| **Retrieval service** | `app/services/retrieval.py`, `reranker.py` | Dense + native sparse → server-side RRF → cross-encoder rerank → top-K + score floor. |
| **Router (agent brain)** | `app/services/router.py` | Decision tree: DIRECT_LLM / RAG / WEB, with explainable reasons. |
| **Generation service** | `app/services/generation.py` | Groq grounded answers, vision captioning, router tie-break, tokens. |
| **Web tool** | `app/services/websearch.py` | Live public search fallback. |
| **Orchestrator** | `app/services/agent.py` | Ties routing → execution → generation → trace persistence. |
| **Evaluation** | `app/services/evaluation.py` | Offline quality/latency/cost harness. |
| **Observability** | `app/core/observability.py`, `api/routes_analytics.py` | Per-step timing/memory/tokens/cost; fleet analytics. |
| **Metadata/ACL DB** | `app/models/db.py` | Users, tenants, documents, versions, jobs, feedback, traces. |

---

## 5. Multi-tenant data model

```
Tenant (id, name, plan, monthly_token_quota)
  └── User (id, tenant_id, workspace_id, email, password_hash, role)
  └── Document (id, tenant_id, workspace_id, user_id, filename,
                object_key, content_hash, version, status,
                n_chunks/n_tables/n_images)
        └── Job (id, document_id, tenant_id, status, stage,
                 attempts, max_attempts, error)
  └── Feedback (id, tenant_id, user_id, request_id, vote, note)
  └── TraceRecord (id, request_id, tenant_id, route, model,
                   tokens, latency_ms, cost_usd, payload[steps])
```

Qdrant payloads **denormalize** the ACL fields
(`tenant_id, workspace_id, document_id, modality, page`) so retrieval can filter
by them at query time. **One collection for all tenants** (payload partitioning)
— never a collection per user (the anti-pattern the assignment calls out).

---

## 6. Async ingestion workflow

```
Upload ──► store object ──► create Document(status=uploaded) + Job(status=queued)
   │                                                    │  (API returns immediately)
   └──────────────── HTTP 200 (non-blocking) ◄──────────┘
                                                        │
        Worker claims job ──► parsing ──► chunking ──► indexing ──► ready
                                 │                                    ▲
                                 └── on error: attempts<max ? requeue │
                                                : status=failed ──────┘
```

Document status transitions: `uploaded → parsing → chunking → indexing → ready`
(or `failed`). The UI polls and shows the live state. Retries with a bounded
`attempts/max_attempts` counter mean transient failures self-heal.

---

## 7. The agentic decision tree (LLM / RAG / Web)

`app/services/router.py` — a **transparent, auditable** tree (not a black box).
It extracts cheap deterministic signals, runs a **retrieval probe**, then decides:

```
force_route set? ───────────────────────────────► honor override
        │no
no corpus? ──► recency cue? ──► WEB
        │       └─ doc cue?  ──► WEB
        │       └─ else      ──► DIRECT_LLM
        │has corpus
run RETRIEVAL PROBE → top rerank score `conf`
        │
conf ≥ threshold & evidence? ───────────────────► RAG          (strong grounding)
        │no
recency cue? ───────────────────────────────────► WEB          (fresh info)
        │no
doc cue & some evidence? ────────────────────────► RAG (low-confidence, flagged)
        │no
else ───────────────────────────────────────────► DIRECT_LLM   (model knowledge)
        │
borderline (|conf − threshold| < 0.12)? ─────────► LLM tie-breaker votes
```

Every branch writes its `branch` name, the extracted features, the retrieval
score, and (if used) the LLM vote into the request trace — so the **Chat tab
shows "Why:"** and the **Analytics tab shows the route mix**. This is the feature
that turns "a RAG demo" into "an agent that reasons about how to answer."

> **Design note:** the *strong-RAG* branch is evaluated **before** the recency
> branch. So a question like *"revenue in 2024"* (which contains a year → recency
> cue) still routes to **RAG** when the documents actually contain the answer;
> it only falls through to WEB when retrieval is genuinely weak.

---

## 8. Hybrid retrieval pipeline — step-by-step reasoning

`app/services/retrieval.py`. Each stage exists to fix a specific failure mode,
and each is individually timed/measured in the trace:

| Stage | Runs where | Problem it solves |
|---|---|---|
| **1. Dense (Qdrant, cosine)** | Qdrant ANN, named vector `dense` | Semantic match: *"top line"* ≈ *"revenue"*. Misses rare exact tokens. |
| **2. Sparse (Qdrant, IDF)** | Qdrant sparse index, named vector `sparse` | Exact lexical match for IDs/SKUs/proper nouns. Values are term frequencies; Qdrant applies **IDF server-side** → BM25-style scoring in the DB. |
| **3. RRF fusion** | **Qdrant, server-side** | ONE `query_points` call fuses the two prefetches with Reciprocal Rank Fusion. (Falls back to in-process RRF only if the backend can't fuse.) |
| **4. Cross-encoder rerank** | reranker model | Re-reads (query, chunk) jointly → precise ordering of the ~40 fused candidates. |
| **5. Top-K + score floor** | service | Only strong evidence reaches the LLM; the floor doubles as the router's RAG-confidence signal. |

Stages 1–3 now happen in a **single Qdrant round trip** using native dense +
sparse vectors on the same points — no client-side BM25 scan over the corpus, so
retrieval scales with Qdrant's index rather than with corpus size. The **score
floor** does double duty: it prevents the generator from being handed junk
context, *and* it's the exact number the router uses to decide RAG vs WEB/DIRECT.

---

## 9. Multimodal design (images + tables)

Going **beyond OCR/Markdown-only** (assignment 3.7 / 3.8):

**Images** (`parsing.py` + `generation.caption_image`):
`Image → Groq vision caption → embed the caption → Qdrant`. Images become
first-class searchable objects via their vision summary; raw bytes are kept in
object storage for source preview. *Advanced path (documented):* native visual /
late-interaction retrieval (ColPali-style) — embed image patches directly and use
multivector late interaction, no captioning bottleneck.

**Tables** (`parsing.py`) — stored in **four** representations so both semantic
*and* exact-fact questions work:
1. `raw` — literal cell grid, for exact "what's the value in row X, col Y" lookups.
2. `markdown` — what the LLM reasons over.
3. `headers/schema` — column names/roles.
4. `summary` — natural-language description used for *retrieval matching*.

Tables and images are **never chunk-split** — they're atomic units, so structure
and captions stay intact.

---

## 10. Security & multi-tenancy

- **AuthN:** stateless JWT (HS256); bcrypt-hashed passwords. `JWT_SECRET` comes
  from a secret manager in prod, never source.
- **AuthZ in the query, not the UI:** `VectorStore.search()` *requires* a
  `Principal` and always injects `tenant_id == principal.tenant_id` (+ workspace)
  into the Qdrant filter. A UI bug **cannot** leak cross-tenant data.
- **Tenant isolation:** single collection, payload-partitioned; payload indexes
  on ACL fields keep filtering fast.
- **Auditability:** every request persists a `TraceRecord` (who, what route, cost,
  latency); feedback is stored per `request_id`.
- **Secret management:** all secrets via env / `.env` (gitignored) → Vault/AWS SM
  in prod. No secrets in URLs or logs.

---

## 11. Document versioning & incremental ingestion

`content_hash = sha256(bytes)`. On upload:
- **Unchanged** (same filename + same hash + already `ready`) → **skip** parsing
  and embedding entirely. No wasted compute.
- **Changed** → bump `version`, and `delete_document()` removes only *that*
  document's vectors before re-indexing — the rest of the workspace is untouched.

---

## 12. Evaluation & observability (the Analytics tab)

**Evaluation** (`services/evaluation.py`, Eval tab): a repeatable dataset of
`{question, expected_source, expected_answer}` yields retrieval recall,
groundedness, answer correctness, citation correctness, avg latency, avg cost,
and failure rate — so model/prompt/chunking/retrieval changes are comparable
across releases. **Two interchangeable judge backends** behind one report shape:
- `proxy` — deterministic lexical metrics (token-F1/overlap). Free, fast,
  CI-stable. Use in CI to A/B a change.
- `llm` — **LLM-as-judge** (Groq): a strong model reads question + evidence +
  answer + reference and scores groundedness/correctness/citation on 0..1 with
  written reasoning. Far better human correlation; costs tokens. Downgrades to
  `proxy` per-item if no key, and the report says which judge actually ran.

**Observability** (`core/observability.py`, Analytics tab): every request is a
timed, memory-sampled trace. The deep-dive tab shows:
- **Latency:** p50 / p95 / avg / max, latency-over-time trend, **per-stage** bar chart (find the bottleneck).
- **Memory:** live process RSS + **per-stage peak heap** (memory hotspots).
- **Routing:** DIRECT/RAG/WEB distribution pie.
- **Cost & tokens:** total tokens and $ (per-model Groq pricing).
- **Quality:** thumbs up/down satisfaction.
- **Subsystem health:** LLM live vs stub, embeddings/reranker model vs fallback —
  so you know *why* the numbers look the way they do.
- **Raw traces:** step-level drill-down table.

---

## 13. Project structure

```
agentic-multimodal-rag/
├── app/                        # FastAPI backend (service-oriented)
│   ├── main.py                 # API gateway: wires routers, CORS, DB init
│   ├── core/
│   │   ├── config.py           # typed settings (one place for every knob)
│   │   ├── security.py         # JWT + Principal (tenant/user/workspace/role)
│   │   ├── observability.py    # Trace: per-step latency/memory/tokens/cost
│   │   └── logging.py
│   ├── models/
│   │   ├── db.py               # SQLAlchemy metadata/ACL model + trace store
│   │   └── schemas.py          # Pydantic request/response contracts
│   ├── services/
│   │   ├── parsing.py          # multimodal PDF: text/table/image extraction
│   │   ├── embeddings.py       # local embeddings + offline hashing fallback
│   │   ├── vectorstore.py      # Qdrant hybrid (dense + NATIVE sparse), ACL-filtered
│   │   ├── sparse.py           # sparse (BM25/IDF) encoder for native hybrid search
│   │   ├── ingestion.py        # object store → parse → chunk → embed(dense+sparse) → index
│   │   ├── reranker.py         # cross-encoder rerank (+ fallback)
│   │   ├── retrieval.py        # dense + sparse → RRF (server-side) → rerank → top-K
│   │   ├── websearch.py        # DuckDuckGo tool
│   │   ├── router.py           # AGENTIC DECISION TREE
│   │   ├── generation.py       # Groq LLM + vision + router classifier + LLM judge
│   │   ├── agent.py            # orchestrator (route → execute → generate → trace)
│   │   └── evaluation.py       # offline eval harness (proxy + LLM-as-judge)
│   └── api/                    # auth / documents / chat / analytics / eval routes
├── worker/ingestion_worker.py  # standalone async job consumer (prod)
├── frontend/                   # Streamlit app (admin/demo) + API client
├── frontend-next/              # React/Next.js production frontend (TS + Tailwind + Recharts)
├── scripts/smoke_test.py       # end-to-end test (no keys/models needed)
├── tests/test_router.py        # router unit tests
├── data/eval_dataset.json      # sample eval set
├── docs/ARCHITECTURE.md, DEPLOYMENT.md
├── docker-compose.yml          # qdrant + postgres + api + worker + frontend
├── Dockerfile.backend / Dockerfile.frontend
├── requirements.txt · Makefile · .env.example
```

---

## 14. Quickstart (local)

**Prereqs:** Python 3.11+. A [free Groq API key](https://console.groq.com/keys)
(the app still runs without one — it uses an offline stub + hashing fallback so
you can see the whole pipeline).

```bash
cd agentic-multimodal-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your GROQ_API_KEY
```

Run the three processes (three terminals), then open http://localhost:8501:

```bash
make api        # FastAPI at :8000  (docs at /docs)
```
```bash
make worker     # async ingestion worker
```
```bash
make frontend   # Streamlit UI at :8501
```

**Prefer the React/Next.js UI?** (needs Node 18+)

```bash
cd frontend-next && cp .env.local.example .env.local && npm install && npm run dev
```
Then open http://localhost:3000. See [frontend-next/README.md](frontend-next/README.md).

Verify without any keys/models:

```bash
PYTHONPATH=. python scripts/smoke_test.py   # end-to-end
pytest -q                                    # router unit tests
```

**First run downloads** the embedding + reranker models (~120 MB). Offline? Set
`ALLOW_EMBEDDING_FALLBACK=true` (default) and it uses a hashing embedder so
everything still works.

---

## 15. Production deployment

Full stack (Qdrant + Postgres + API + 2 workers + frontend):

```bash
cp .env.example .env    # set GROQ_API_KEY and a strong JWT_SECRET
docker compose up --build
```

Compose already: points the API/worker at **server Qdrant** and **Postgres**,
mounts shared **object storage**, and runs **2 worker replicas** independent of
the API. See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the hardening
checklist (S3/MinIO, managed queue, TLS ingress, secrets manager, autoscaling,
backups, rate limits/quotas).

---

## 16. Phased roadmap → code map

The assignment's 9 phases, mapped to where each lives in this repo:

| Phase | Assignment goal | Where it lives here |
|---|---|---|
| 1 · V1 prototype | parse/index/dense/generate | `services/parsing,embeddings,generation` |
| 2 · Multi-doc + auth | identity, ownership | `core/security`, `models/db`, `api/routes_auth` |
| 3 · Async + storage + PG | workers, object store, Postgres | `worker/`, `ingestion.store_object`, `docker-compose` |
| 4 · Tenant/RBAC filtering | ACL in retrieval | `vectorstore._acl_filter`, `Principal` |
| 5 · Hybrid + rerank | dense+sparse+RRF+rerank | `services/retrieval`, `reranker` |
| 6 · Better multimodal | vision captions, table intelligence | `parsing.py`, `generation.caption_image` |
| 7 · Eval + observability | evals, tracing, dashboards | `services/evaluation`, `core/observability`, Analytics tab |
| 8 · Production API + FE | clean APIs, scalable deploy | `app/api/*`, Docker, compose |
| 9 · Enterprise + billing | quotas, audit, connectors | `Tenant.plan/quota`, `Feedback`, `TraceRecord` (audit); connectors = next |

**Build-first priority & why:** service separation + auth + ACL-filtered
retrieval come first — they're the load-bearing walls; retrieval quality,
multimodal depth, and evaluation are the finish work layered on a sound
structure. You can't safely add tenants to a monolith after the fact.
