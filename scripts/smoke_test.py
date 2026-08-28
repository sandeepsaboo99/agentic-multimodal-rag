"""End-to-end smoke test with embedded Qdrant + fallback embeddings + offline LLM.

Exercises: register -> upload PDF -> (inline) ingest -> chat (RAG route) ->
analytics summary. No Groq key or heavy models required.
"""
import io
import os
import tempfile

os.environ.setdefault("ALLOW_EMBEDDING_FALLBACK", "true")
# isolate storage so repeat runs are clean
_tmp = tempfile.mkdtemp()
os.environ["QDRANT_LOCAL_PATH"] = os.path.join(_tmp, "qdrant")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_tmp, 'test.db')}"
os.environ["OBJECT_STORAGE_PATH"] = os.path.join(_tmp, "objects")
os.environ["ROUTER_USE_LLM"] = "false"

import fitz  # PyMuPDF
from fastapi.testclient import TestClient

from app.main import app
from app.services.ingestion import run_ingestion_job
from app.models.db import Job, session_scope

client = TestClient(app)


def make_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), (
        "Acme Corporation Annual Report 2024.\n"
        "The primary goal of Acme is sustainable energy storage.\n"
        "Total revenue for fiscal year 2024 was 42 million dollars.\n"
        "Research and development spending increased by 18 percent.\n"
    ), fontsize=11)
    return doc.tobytes()


def main() -> None:
    # 1. register
    r = client.post("/auth/register", json={"email": "a@acme.com", "password": "pw12345", "tenant_name": "Acme"})
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    H = {"Authorization": f"Bearer {token}"}
    print("register: OK")

    # 2. upload (BackgroundTasks won't run under TestClient reliably, so run job inline)
    r = client.post("/documents/upload", files={"file": ("report.pdf", make_pdf(), "application/pdf")}, headers=H)
    assert r.status_code == 200, r.text
    print("upload:", r.json()["status"])

    with session_scope() as db:
        job_id = db.query(Job).order_by(Job.created_at.desc()).first().id
    run_ingestion_job(job_id)

    docs = client.get("/documents", headers=H).json()
    print("documents:", [(d["filename"], d["status"], d["n_chunks"]) for d in docs])
    assert docs[0]["status"] == "ready"

    # 3. chat — should route to RAG and cite the document
    r = client.post("/chat", json={"query": "What was total revenue in 2024?", "history": []}, headers=H)
    assert r.status_code == 200, r.text
    resp = r.json()
    print("route:", resp["route"], "| reason:", resp["route_reason"])
    print("citations:", [(c["filename"], c["page"], c["score"]) for c in resp["citations"]])
    print("answer preview:", resp["answer"][:120])
    print("trace steps:", [s["name"] for s in resp["trace"]["steps"]])

    # 4. chit-chat — should route DIRECT_LLM (greetings never need retrieval)
    r2 = client.post("/chat", json={"query": "hello there", "history": []}, headers=H)
    print("chitchat route:", r2.json()["route"])
    assert r2.json()["route"] == "DIRECT_LLM", f"expected DIRECT_LLM, got {r2.json()['route']}"

    # 5. analytics
    a = client.get("/analytics/summary", headers=H).json()
    print("analytics route_mix:", a["route_mix"], "| p50:", a["latency_ms"]["p50"], "ms")
    print("stage latency:", a["stage_latency_avg_ms"])

    # 6. evaluation (proxy judge — no key needed; llm judge falls back to proxy)
    ev = client.post("/eval/run", json={
        "items": [{"question": "What was total revenue in 2024?",
                   "expected_source": "report.pdf", "expected_answer": "42 million dollars"}],
        "judge": "llm",
    }, headers=H)
    assert ev.status_code == 200, ev.text
    rep = ev.json()
    print("eval judge:", rep["judge"], "| recall:", rep["retrieval_recall"],
          "| groundedness:", rep["groundedness"])
    assert rep["n"] == 1 and rep["judge"] in ("proxy", "llm")
    print("\nSMOKE TEST PASSED ✅")


if __name__ == "__main__":
    main()
