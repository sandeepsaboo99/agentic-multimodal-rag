# Production Deployment Guide

This covers taking InsightRAG from `docker compose up` on a laptop to a
hardened, scalable deployment. Ordered by what to do first.

## 0. Topology

```
        Internet
           │  TLS
   ┌───────▼────────┐
   │ Ingress / ALB  │  (TLS termination, WAF, rate limits)
   └───────┬────────┘
   ┌───────▼────────┐        ┌──────────────┐
   │  API (N pods)  │◄──────►│   Postgres   │ (managed, Multi-AZ)
   │  FastAPI/uvicorn│        └──────────────┘
   └───┬────────┬───┘        ┌──────────────┐
       │        │            │    Qdrant    │ (managed / StatefulSet, replicated)
       │        └───────────►└──────────────┘
   ┌───▼────────────┐        ┌──────────────┐
   │ Queue (SQS/    │        │ Object store │ (S3 / MinIO)
   │  Redis/Rabbit) │        └──────────────┘
   └───┬────────────┘
   ┌───▼────────────┐
   │ Workers (M pods)│  autoscaled on queue depth
   └────────────────┘
```

API pods and worker pods scale **independently** — the whole reason for the
service split. Workers autoscale on **queue depth**; API on **CPU/RPS**.

## 1. Configuration & secrets

- Provide config via environment (12-factor). Never bake secrets into images.
- `JWT_SECRET`, `GROQ_API_KEY`, DB creds → **AWS Secrets Manager / Vault / GCP
  Secret Manager**, injected as env at runtime.
- Rotate `JWT_SECRET` with a key-id (`kid`) header + dual-key verification window.
- Set `allow_origins` in `app/main.py` CORS to your real frontend origin only.

## 2. Storage

| Data | Dev | Production |
|---|---|---|
| PDFs + images | local dir | **S3 / MinIO** (versioned bucket, SSE-KMS) — set `OBJECT_STORAGE_PATH` to a mount or swap `store_object` for the S3 SDK |
| Metadata/ACL | SQLite | **Postgres** (Multi-AZ, PITR backups) — `DATABASE_URL=postgresql+psycopg2://…` |
| Vectors | embedded Qdrant | **Qdrant server/cluster** — `QDRANT_URL=https://…`, `QDRANT_API_KEY=…`, enable payload indexes + replication |

Run Alembic migrations instead of `create_all` for schema evolution in prod.

## 3. Job queue

Replace `worker.claim_next_job` (DB-poll) with a real broker:
- **SQS** (managed, at-least-once) or **Redis Streams** / **RabbitMQ**.
- Keep the `attempts/max_attempts` retry + a **dead-letter queue** for poison jobs.
- Make ingestion **idempotent** (it already deletes-then-reindexes per doc), so
  at-least-once redelivery is safe.

## 4. Scaling & performance

- **API:** stateless → scale horizontally behind the load balancer. `uvicorn
  --workers` per pod; add `gunicorn -k uvicorn.workers.UvicornWorker` for process
  management.
- **Workers:** CPU-bound (embeddings). Autoscale on queue depth; pin `torch`
  threads; consider a GPU node pool for the reranker/embeddings at scale.
- **Embeddings/reranker:** load models once per process (already singletons).
  For heavy load, extract them into a dedicated model-serving service (TEI /
  Triton) so API/worker pods stay light.
- **Qdrant:** size RAM to the vector count; enable quantization for large corpora.

## 5. Reliability

- **Health checks:** `/health` (liveness) — add a `/ready` that pings Qdrant + DB.
- **Timeouts & retries** on every outbound call (Groq, Qdrant, web).
- **Circuit breaker** around Groq; the app already degrades to an offline stub.
- **Graceful shutdown:** drain in-flight jobs on SIGTERM.

## 6. Observability in prod

The built-in trace store is great for the demo tab; in prod also:
- Emit **OpenTelemetry** spans from `Trace.step` → Tempo/Jaeger.
- Ship structured logs → Loki/CloudWatch; metrics (latency, route mix, cost,
  queue depth) → Prometheus/Grafana with alerts on p95 latency and failure rate.
- Sample full traces (payloads can be large); aggregate the rest.

## 7. Cost & quota controls

- `Tenant.monthly_token_quota` + `TraceRecord` cost give you per-tenant metering.
- Enforce quotas at the API gateway; return 429 when exceeded.
- Cache identical (tenant, query) answers with a short TTL to cut Groq spend.

## 8. Security hardening checklist

- [ ] TLS everywhere; HSTS at the edge.
- [ ] Secrets in a manager, not env files; images scanned (Trivy).
- [ ] Least-privilege IAM for S3/queue.
- [ ] Rate limiting + WAF; request size caps on upload.
- [ ] Input validation on PDFs (size, page count, MIME sniffing).
- [ ] Audit log retention policy (TraceRecord + Feedback).
- [ ] Pen-test the ACL filter: attempt cross-tenant retrieval and confirm 0 leaks.

## 9. CI/CD

- `pytest -q` + `python scripts/smoke_test.py` on every PR (no keys needed).
- Run the eval harness nightly against a golden dataset; block releases on
  regression in recall/groundedness/cost.
- Build + scan images; deploy via blue/green or rolling with health gates.
