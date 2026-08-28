"""
InsightRAG — Streamlit frontend.

This is the demo/admin UI of the target architecture. It is a *thin view layer*:
it holds no business logic and reaches the platform only through the FastAPI
backend via api_client. Tabs:

  1. Chat          - the agentic answer path, showing the ROUTE the agent chose,
                     WHY (decision-tree reasoning), citations, and a per-step
                     trace (timing + memory) for that single request.
  2. Documents     - async upload with live status (uploaded->...->ready).
  3. Analytics     - the deep-dive performance tab: latency percentiles, route
                     mix, per-stage latency/memory, token + $ cost, feedback,
                     live process memory, and raw trace drill-down.
  4. Evaluation    - run the offline eval harness and see quality metrics.

Run:  streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import api_client as api

st.set_page_config(page_title="InsightRAG", page_icon="🧠", layout="wide")

# ---- session state ----
for k, v in {"token": None, "identity": None, "history": [], "last_response": None}.items():
    st.session_state.setdefault(k, v)


# ============================ SIDEBAR: AUTH ==================================
def auth_sidebar() -> None:
    st.sidebar.title("🧠 InsightRAG")
    st.sidebar.caption("Agentic Multimodal RAG — production demo")

    if st.session_state.token:
        idn = st.session_state.identity
        st.sidebar.success(f"Signed in\n\nrole: **{idn['role']}**")
        st.sidebar.code(f"tenant={idn['tenant_id'][:8]}…\nuser={idn['user_id'][:8]}…", language=None)
        if st.sidebar.button("Sign out"):
            for k in ("token", "identity", "history", "last_response"):
                st.session_state[k] = None if k != "history" else []
            st.rerun()
        return

    mode = st.sidebar.radio("Access", ["Login", "Register"], horizontal=True)
    email = st.sidebar.text_input("Email", value="admin@acme.com")
    password = st.sidebar.text_input("Password", type="password", value="demo1234")
    tenant = st.sidebar.text_input("Organization", value="Acme Inc") if mode == "Register" else None
    if st.sidebar.button(mode, type="primary"):
        try:
            res = api.register(email, password, tenant) if mode == "Register" else api.login(email, password)
            st.session_state.token = res["access_token"]
            st.session_state.identity = res
            st.rerun()
        except api.APIError as e:
            st.sidebar.error(str(e))


# ============================ TAB 1: CHAT ====================================
ROUTE_BADGE = {
    "RAG": "🟢 RAG (your documents)",
    "WEB": "🌐 WEB (live search)",
    "DIRECT_LLM": "💬 DIRECT (model knowledge)",
}


def chat_tab() -> None:
    st.subheader("Agentic Chat")
    st.caption("The agent decides — per question — whether to use your documents (RAG), "
               "live web search (WEB), or answer directly (DIRECT_LLM).")

    colq, colr = st.columns([4, 1])
    force = colr.selectbox("Force route", ["auto", "RAG", "WEB", "DIRECT_LLM"],
                           help="Override the decision tree to compare behaviours.")
    query = colq.text_input("Ask a question", key="chat_input",
                            placeholder="e.g. What was total revenue in the Q3 table?")

    if st.button("Send", type="primary") and query:
        with st.spinner("Agent is routing and answering…"):
            try:
                res = api.chat(st.session_state.token, query, st.session_state.history,
                               None if force == "auto" else force)
                st.session_state.last_response = res
                st.session_state.history.append({"role": "user", "content": query})
                st.session_state.history.append({"role": "assistant", "content": res["answer"]})
            except api.APIError as e:
                st.error(str(e))

    res = st.session_state.last_response
    if not res:
        return

    st.markdown(f"**Route chosen:** {ROUTE_BADGE.get(res['route'], res['route'])}")
    st.info(f"🧭 **Why:** {res['route_reason']}")
    st.markdown("### Answer")
    st.write(res["answer"])

    if res["citations"]:
        st.markdown("### Sources")
        for i, c in enumerate(res["citations"], 1):
            loc = f"p.{c['page']}" if c.get("page") else c.get("modality", "")
            with st.expander(f"[{i}] {c['filename']} — {c['modality']} {loc} (score {c['score']})"):
                st.write(c["snippet"])

    # feedback
    fb1, fb2, _ = st.columns([1, 1, 6])
    if fb1.button("👍 Helpful"):
        api.feedback(st.session_state.token, res["request_id"], 1)
        st.toast("Thanks for the feedback!")
    if fb2.button("👎 Not helpful"):
        api.feedback(st.session_state.token, res["request_id"], -1)
        st.toast("Recorded — we'll use it to improve.")

    # per-request trace (timing + memory) — the technical drill-down
    with st.expander("🔬 Technical trace for THIS request (timing + memory per step)"):
        steps = res["trace"]["steps"]
        if steps:
            df = pd.DataFrame(steps)
            st.dataframe(
                df[["name", "latency_ms", "heap_peak_mb", "rss_delta_mb"]]
                .rename(columns={"name": "step", "latency_ms": "latency (ms)",
                                 "heap_peak_mb": "peak heap (MB)", "rss_delta_mb": "RSS Δ (MB)"}),
                use_container_width=True, hide_index=True,
            )
            fig = px.bar(df, x="name", y="latency_ms", title="Per-step latency (ms)",
                         labels={"name": "pipeline step", "latency_ms": "ms"})
            st.plotly_chart(fig, use_container_width=True)
        st.json({"tokens": res["trace"]["total_tokens"], "model": res["trace"]["model"],
                 "cost_usd": res["trace"].get("cost_usd"),
                 "total_latency_ms": res["trace"]["total_latency_ms"]})
        st.markdown("**Router feature signals**")
        route_step = next((s for s in steps if s["name"] == "route_decision"), None)
        if route_step:
            st.json(route_step["meta"])


# ============================ TAB 2: DOCUMENTS ===============================
STATUS_ICON = {"uploaded": "📤", "parsing": "📖", "chunking": "✂️", "indexing": "🧩",
               "ready": "✅", "failed": "❌"}


def documents_tab() -> None:
    st.subheader("Documents")
    st.caption("Upload returns immediately; parsing/embedding runs asynchronously. "
               "Watch each doc move through uploaded → parsing → chunking → indexing → ready.")

    files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if st.button("Ingest", type="primary") and files:
        for f in files:
            try:
                api.upload(st.session_state.token, f.name, f.read())
                st.success(f"Queued: {f.name}")
            except api.APIError as e:
                st.error(f"{f.name}: {e}")

    if st.button("🔄 Refresh status"):
        st.rerun()

    try:
        docs = api.list_documents(st.session_state.token)
    except api.APIError as e:
        st.error(str(e))
        return
    if not docs:
        st.info("No documents yet. Upload a PDF to enable RAG.")
        return

    for d in docs:
        c1, c2, c3, c4 = st.columns([4, 2, 3, 1])
        c1.markdown(f"{STATUS_ICON.get(d['status'], '•')} **{d['filename']}** (v{d['version']})")
        c2.markdown(f"`{d['status']}`")
        c3.caption(f"{d['n_chunks']} text · {d['n_tables']} tables · {d['n_images']} images")
        if c4.button("🗑", key=f"del_{d['id']}"):
            api.delete_document(st.session_state.token, d["id"])
            st.rerun()
        if d.get("error"):
            st.error(d["error"])


# ============================ TAB 3: ANALYTICS ===============================
def analytics_tab() -> None:
    st.subheader("📊 Performance Deep-Dive")
    st.caption("Operational + quality telemetry aggregated from every request's trace.")

    try:
        s = api.analytics_summary(st.session_state.token)
        health = api.analytics_health(st.session_state.token)
    except api.APIError as e:
        st.error(str(e))
        return

    # subsystem health strip — explains WHY numbers look the way they do
    hc = st.columns(4)
    hc[0].metric("LLM", "live" if health["llm_available"] else "offline-stub")
    hc[1].metric("Embeddings", "fallback" if health["embedding_fallback"] else "model")
    hc[2].metric("Reranker", "fallback" if health["reranker_fallback"] else "cross-encoder")
    hc[3].metric("Process RSS", f"{s['process_memory_mb']} MB")

    if s["requests"] == 0:
        st.info("No requests yet — ask something in the Chat tab to populate analytics.")
        return

    # KPI row
    k = st.columns(5)
    k[0].metric("Requests", s["requests"])
    k[1].metric("p50 latency", f"{s['latency_ms']['p50']} ms")
    k[2].metric("p95 latency", f"{s['latency_ms']['p95']} ms")
    k[3].metric("Tokens", f"{s['tokens_total']:,}")
    k[4].metric("Cost", f"${s['cost_usd_total']:.4f}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Route distribution** (how often the agent used RAG vs WEB vs LLM)")
        if s["route_mix"]:
            rm = pd.DataFrame({"route": list(s["route_mix"]), "count": list(s["route_mix"].values())})
            st.plotly_chart(px.pie(rm, names="route", values="count", hole=0.45),
                            use_container_width=True)
    with c2:
        st.markdown("**Average latency by pipeline stage** (find the bottleneck)")
        sl = s["stage_latency_avg_ms"]
        if sl:
            df = pd.DataFrame({"stage": list(sl), "avg_ms": list(sl.values())}).sort_values("avg_ms")
            st.plotly_chart(px.bar(df, x="avg_ms", y="stage", orientation="h"),
                            use_container_width=True)

    st.markdown("**Peak heap by pipeline stage (MB)** — memory hotspots")
    sm = s["stage_heap_peak_avg_mb"]
    if sm:
        df = pd.DataFrame({"stage": list(sm), "peak_MB": list(sm.values())}).sort_values("peak_MB")
        st.plotly_chart(px.bar(df, x="peak_MB", y="stage", orientation="h"),
                        use_container_width=True)

    fb = s["feedback"]
    st.markdown(f"**User feedback:** 👍 {fb['up']}  ·  👎 {fb['down']}  ·  "
                f"satisfaction: {fb['score'] if fb['score'] is not None else '—'}")

    st.markdown("### Recent request traces")
    traces = api.analytics_traces(st.session_state.token, 50)
    if traces:
        tdf = pd.DataFrame([{
            "time": t["created_at"][11:19], "route": t["route"],
            "latency_ms": t["total_latency_ms"], "tokens": t["prompt_tokens"] + t["completion_tokens"],
            "cost_usd": t["cost_usd"], "query": t["query"][:60],
        } for t in traces])
        st.dataframe(tdf, use_container_width=True, hide_index=True)

        # latency-over-time trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=tdf["latency_ms"][::-1].reset_index(drop=True),
                                 mode="lines+markers", name="latency ms"))
        fig.update_layout(title="Latency trend (oldest → newest)", height=300)
        st.plotly_chart(fig, use_container_width=True)


# ============================ TAB 4: EVALUATION ==============================
DEFAULT_EVAL = [
    {"question": "What is the main topic of the document?", "expected_source": "", "expected_answer": ""},
]


def evaluation_tab() -> None:
    st.subheader("🧪 Evaluation Harness")
    st.caption("Run a repeatable eval set (question / expected source / expected answer) and "
               "measure retrieval recall, groundedness, correctness, latency and cost.")

    judge = st.radio(
        "Judge backend", ["proxy", "llm"], horizontal=True,
        help="proxy = deterministic lexical metrics (free, CI-stable). "
             "llm = LLM-as-judge via Groq (needs GROQ_API_KEY; higher quality).",
    )
    txt = st.text_area(
        "Eval dataset (JSON list of {question, expected_source, expected_answer})",
        value=json.dumps(DEFAULT_EVAL, indent=2), height=200,
    )
    if st.button("Run evaluation", type="primary"):
        try:
            items = json.loads(txt)
        except Exception as e:  # noqa: BLE001
            st.error(f"Invalid JSON: {e}")
            return
        with st.spinner(f"Running eval ({judge} judge) across the agent…"):
            try:
                rep = api.run_eval(st.session_state.token, items, judge)
            except api.APIError as e:
                st.error(str(e))
                return
        st.caption(f"Judge used: **{rep['judge']}**"
                   + ("" if rep["judge"] == judge else "  (fell back — no Groq key)"))
        m = st.columns(4)
        m[0].metric("Retrieval recall", rep["retrieval_recall"])
        m[1].metric("Groundedness", rep["groundedness"])
        m[2].metric("Answer correctness", rep["answer_correctness"])
        m[3].metric("Citation correctness", rep["citation_correctness"])
        m2 = st.columns(3)
        m2[0].metric("Avg latency", f"{rep['avg_latency_ms']} ms")
        m2[1].metric("Avg cost", f"${rep['avg_cost_usd']:.5f}")
        m2[2].metric("Failure rate", rep["failure_rate"])
        st.dataframe(pd.DataFrame(rep["details"]), use_container_width=True, hide_index=True)


# ============================ MAIN ===========================================
def main() -> None:
    auth_sidebar()
    if not st.session_state.token:
        st.title("InsightRAG")
        st.markdown(
            "Agentic **Multimodal RAG** with a smart LLM / RAG / Web decision tree, "
            "hybrid retrieval, and a performance deep-dive.\n\n"
            "👈 **Register or log in** to begin.")
        return

    t1, t2, t3, t4 = st.tabs(["💬 Chat", "📄 Documents", "📊 Analytics", "🧪 Evaluation"])
    with t1:
        chat_tab()
    with t2:
        documents_tab()
    with t3:
        analytics_tab()
    with t4:
        evaluation_tab()


main()
