"""
Agentic decision-tree router: choose DIRECT_LLM vs RAG vs WEB.

Design reasoning (the user's headline feature)
----------------------------------------------
A naive RAG app always retrieves — wasting latency/tokens on "hi", and
hallucinating from irrelevant chunks on out-of-corpus questions. A production
agent instead *decides how to answer*. This router is a transparent, auditable
DECISION TREE, not a black box:

  Feature extraction (cheap, deterministic signals):
    - temporal/recency cues  -> ("today", "latest", "current", years, "news"...)
    - private-doc cues        -> ("the document", "this pdf", "table", "figure"...)
    - conversational cues     -> greetings / meta questions
    - corpus availability      -> does this tenant even have indexed docs?
    - retrieval PROBE score    -> we run the real hybrid retriever and read the
                                  reranker's top score as an evidence-strength signal.

  Decision tree (evaluated top-down, first match wins):
    1. force_route set by caller          -> honor it (manual override).
    2. no corpus AND recency cue          -> WEB
    3. no corpus                          -> DIRECT_LLM
    4. strong retrieval (score >= conf)   -> RAG
    5. recency cue AND weak retrieval      -> WEB
    6. private-doc cue but weak retrieval  -> RAG (best effort) but flagged low-confidence
    7. otherwise                          -> DIRECT_LLM

  LLM tie-breaker (optional): for ambiguous cases (steps 4-7 near the threshold)
  a cheap 8B model casts a vote; the tree records both its own decision and the
  LLM's, and the reason string explains exactly which branch fired. This makes
  every routing decision explainable in the analytics tab.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.observability import Trace
from app.core.security import Principal
from app.services.generation import get_generation_service
from app.services.retrieval import Evidence, get_retrieval_service

log = get_logger(__name__)

_RECENCY = re.compile(
    r"\b(today|tonight|yesterday|now|current(ly)?|latest|recent|breaking|news|"
    r"price|stock|weather|forecast|score|live|update|this (week|month|year)|"
    r"20\d{2}|as of)\b",
    re.I,
)
_DOC_CUE = re.compile(
    r"\b(document|doc|pdf|report|file|uploaded|attachment|table|figure|chart|"
    r"image|page|section|appendix|according to|in the (paper|report|doc))\b",
    re.I,
)
_CHITCHAT = re.compile(
    r"^\s*(hi|hey|hello|thanks|thank you|good (morning|evening)|how are you|"
    r"who are you|what can you do)\b",
    re.I,
)


@dataclass
class RouteDecision:
    route: str  # DIRECT_LLM | RAG | WEB
    reason: str
    features: dict = field(default_factory=dict)
    rag_confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)


class AgenticRouter:
    def __init__(self) -> None:
        self.s = get_settings()

    def _features(self, query: str, has_corpus: bool) -> dict:
        return {
            "has_corpus": has_corpus,
            "recency_cue": bool(_RECENCY.search(query)),
            "doc_cue": bool(_DOC_CUE.search(query)),
            "chitchat": bool(_CHITCHAT.search(query)),
            "length": len(query.split()),
        }

    def decide(
        self,
        principal: Principal,
        query: str,
        trace: Trace,
        has_corpus: bool,
        force_route: str | None = None,
        document_ids: list[str] | None = None,
    ) -> RouteDecision:
        with trace.step("route_decision") as m:
            feats = self._features(query, has_corpus)

            # (1) manual override
            if force_route:
                m.update(feats, branch="override")
                return RouteDecision(force_route, f"Manual override -> {force_route}", feats)

            # (1b) greetings / meta questions never need retrieval or web
            if feats["chitchat"] and not feats["doc_cue"]:
                m.update(feats, branch="chitchat_direct")
                return RouteDecision(
                    "DIRECT_LLM", "Conversational/greeting -> answer directly (no retrieval needed).",
                    feats)

            # (2/3) no corpus
            if not has_corpus:
                if feats["recency_cue"]:
                    m.update(feats, branch="no_corpus_recency")
                    return RouteDecision("WEB", "No indexed documents + recency cue -> web search.", feats)
                if not feats["chitchat"] and feats["doc_cue"]:
                    m.update(feats, branch="no_corpus_doc_cue")
                    return RouteDecision(
                        "WEB", "No indexed documents yet, but question expects sources -> web.", feats)
                m.update(feats, branch="no_corpus_direct")
                return RouteDecision("DIRECT_LLM", "No indexed documents -> answer directly.", feats)

            # Retrieval PROBE — the tree's most important signal.
            evidence, conf = get_retrieval_service().retrieve(principal, query, trace, document_ids)
            feats["retrieval_top_score"] = round(conf, 4)
            feats["n_evidence"] = len(evidence)

            # (4) strong evidence -> RAG
            if conf >= self.s.router_rag_confidence and evidence:
                m.update(feats, branch="strong_rag")
                dec = RouteDecision("RAG", f"Strong retrieval (score={conf:.2f}) -> grounded RAG answer.",
                                    feats, conf, evidence)
            # (5) recency + weak evidence -> WEB
            elif feats["recency_cue"]:
                m.update(feats, branch="recency_over_weak_rag")
                dec = RouteDecision("WEB", f"Recency cue with weak retrieval (score={conf:.2f}) -> web.",
                                    feats, conf, evidence)
            # (6) explicit doc reference but weak -> best-effort RAG, low confidence
            elif feats["doc_cue"] and evidence:
                m.update(feats, branch="weak_rag_doc_cue")
                dec = RouteDecision("RAG", f"Question references documents; best-effort RAG "
                                    f"(low confidence {conf:.2f}).", feats, conf, evidence)
            # (7) fallback
            else:
                m.update(feats, branch="fallback_direct")
                dec = RouteDecision("DIRECT_LLM",
                                    f"Weak retrieval (score={conf:.2f}) and no strong doc/recency cue "
                                    f"-> answer directly.", feats, conf, evidence)

            # LLM tie-breaker for borderline cases near the confidence threshold.
            near = abs(conf - self.s.router_rag_confidence) < 0.12
            if self.s.router_use_llm and near:
                vote = get_generation_service().classify_route(query, has_corpus, conf)
                if vote.get("route") in {"DIRECT_LLM", "RAG", "WEB"} and vote["route"] != dec.route:
                    m["llm_vote"] = vote
                    dec = RouteDecision(
                        vote["route"],
                        f"Decision tree said {dec.route}; LLM tie-breaker chose {vote['route']}: "
                        f"{vote.get('reason', '')}",
                        feats, conf, dec.evidence if vote["route"] == "RAG" else [],
                    )
            m["route"] = dec.route
            return dec


_router: AgenticRouter | None = None


def get_router() -> AgenticRouter:
    global _router
    if _router is None:
        _router = AgenticRouter()
    return _router
