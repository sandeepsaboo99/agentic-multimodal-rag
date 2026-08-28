# InsightRAG — React / Next.js Frontend

A production-oriented alternative to the Streamlit demo UI. Same backend, same
API contract — this proves the point that the FastAPI layer is the stable
boundary, so the frontend is swappable (assignment 3.1).

**Stack:** Next.js 14 (App Router) · React 18 · TypeScript · Tailwind CSS · Recharts.

## Tabs
- **Chat** — agentic answers with the chosen route badge (RAG/WEB/DIRECT), the
  decision-tree *reasoning*, citations, thumbs feedback, and a per-request trace
  (latency bar chart + step table).
- **Documents** — async upload that **auto-polls** while docs move through
  `uploaded → parsing → chunking → indexing → ready`.
- **Analytics** — the deep-dive: subsystem health, KPI tiles, route-mix pie,
  per-stage latency + peak-heap bars, latency trend line, feedback, raw traces.
- **Evaluation** — run the eval harness with a **proxy** or **LLM-as-judge** backend.

## Local development

```bash
cd frontend-next
cp .env.local.example .env.local     # point NEXT_PUBLIC_BACKEND_URL at the API
npm install
npm run dev                          # http://localhost:3000
```

The FastAPI backend must be running (see the root README: `make api`). CORS is
open on the backend for local dev.

Type-check without a full build:

```bash
npm run typecheck
```

## Production build

```bash
npm run build && npm start           # standalone Node server on :3000
```

Or via Docker (multi-stage, slim standalone image):

```bash
docker build -t insightrag-web \
  --build-arg NEXT_PUBLIC_BACKEND_URL=https://api.your-domain.com .
docker run -p 3000:3000 insightrag-web
```

> `NEXT_PUBLIC_BACKEND_URL` is read from the **browser**, so set it to the API's
> public URL and note it is inlined at **build** time (rebuild to change it).

## Where things live
```
app/            layout.tsx · page.tsx (auth gate + tab shell) · globals.css
components/      AuthPanel · ChatTab · DocumentsTab · AnalyticsTab · EvalTab · ui
lib/             api.ts (typed client, the only network seam) · types.ts
```
