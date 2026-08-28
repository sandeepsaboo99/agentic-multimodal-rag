"""
Router feature-extraction unit tests.

These test the deterministic parts of the agentic decision tree (feature
extraction + no-corpus branches) without needing models, Qdrant, or Groq — so
they run in CI in milliseconds and lock the routing contract.
"""
from __future__ import annotations

from app.core.observability import Trace
from app.core.security import Principal
from app.services.router import AgenticRouter

PRINCIPAL = Principal("u1", "t1", "default", "member")


def _trace() -> Trace:
    return Trace(request_id="r", query="", tenant_id="t1", user_id="u1")


def test_recency_cue_detected():
    r = AgenticRouter()
    feats = r._features("what is the latest news on AI today?", has_corpus=False)
    assert feats["recency_cue"] is True


def test_doc_cue_detected():
    r = AgenticRouter()
    feats = r._features("according to the uploaded report, what is revenue?", has_corpus=True)
    assert feats["doc_cue"] is True


def test_chitchat_detected():
    r = AgenticRouter()
    feats = r._features("hello there", has_corpus=True)
    assert feats["chitchat"] is True


def test_no_corpus_recency_routes_web():
    r = AgenticRouter()
    dec = r.decide(PRINCIPAL, "what's the current bitcoin price?", _trace(), has_corpus=False)
    assert dec.route == "WEB"


def test_no_corpus_general_routes_direct():
    r = AgenticRouter()
    dec = r.decide(PRINCIPAL, "explain how photosynthesis works", _trace(), has_corpus=False)
    assert dec.route == "DIRECT_LLM"


def test_force_route_override():
    r = AgenticRouter()
    dec = r.decide(PRINCIPAL, "anything", _trace(), has_corpus=False, force_route="WEB")
    assert dec.route == "WEB"


def test_chitchat_routes_direct_even_with_corpus():
    # greetings must never trigger the retrieval probe / RAG, corpus or not
    r = AgenticRouter()
    dec = r.decide(PRINCIPAL, "hello there", _trace(), has_corpus=True)
    assert dec.route == "DIRECT_LLM"
