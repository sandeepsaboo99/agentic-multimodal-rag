"""Thin HTTP client the Streamlit app uses to talk to the FastAPI backend.

Keeping ALL network calls here (not scattered across the UI) keeps the frontend
a pure view layer and makes the frontend/backend boundary explicit — the same
boundary that lets a React/Next.js app later replace Streamlit with no backend
changes (assignment 3.1).
"""
from __future__ import annotations

import os
from typing import Any

import requests

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT = 120


class APIError(Exception):
    pass


def _headers(token: str | None) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _handle(resp: requests.Response) -> Any:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise APIError(f"{resp.status_code}: {detail}")
    return resp.json()


def register(email: str, password: str, tenant_name: str) -> dict:
    return _handle(requests.post(f"{BACKEND}/auth/register",
                   json={"email": email, "password": password, "tenant_name": tenant_name},
                   timeout=TIMEOUT))


def login(email: str, password: str) -> dict:
    return _handle(requests.post(f"{BACKEND}/auth/login",
                   json={"email": email, "password": password}, timeout=TIMEOUT))


def upload(token: str, filename: str, data: bytes) -> dict:
    return _handle(requests.post(f"{BACKEND}/documents/upload",
                   files={"file": (filename, data, "application/pdf")},
                   headers=_headers(token), timeout=TIMEOUT))


def list_documents(token: str) -> list[dict]:
    return _handle(requests.get(f"{BACKEND}/documents", headers=_headers(token), timeout=TIMEOUT))


def delete_document(token: str, doc_id: str) -> dict:
    return _handle(requests.delete(f"{BACKEND}/documents/{doc_id}",
                   headers=_headers(token), timeout=TIMEOUT))


def chat(token: str, query: str, history: list[dict], force_route: str | None) -> dict:
    return _handle(requests.post(f"{BACKEND}/chat",
                   json={"query": query, "history": history, "force_route": force_route},
                   headers=_headers(token), timeout=TIMEOUT))


def feedback(token: str, request_id: str, vote: int, note: str = "") -> dict:
    return _handle(requests.post(f"{BACKEND}/feedback",
                   json={"request_id": request_id, "vote": vote, "note": note},
                   headers=_headers(token), timeout=TIMEOUT))


def analytics_summary(token: str) -> dict:
    return _handle(requests.get(f"{BACKEND}/analytics/summary", headers=_headers(token), timeout=TIMEOUT))


def analytics_traces(token: str, limit: int = 50) -> list[dict]:
    return _handle(requests.get(f"{BACKEND}/analytics/traces?limit={limit}",
                   headers=_headers(token), timeout=TIMEOUT))


def analytics_health(token: str) -> dict:
    return _handle(requests.get(f"{BACKEND}/analytics/health", headers=_headers(token), timeout=TIMEOUT))


def run_eval(token: str, items: list[dict], judge: str = "proxy") -> dict:
    return _handle(requests.post(f"{BACKEND}/eval/run",
                   json={"items": items, "judge": judge},
                   headers=_headers(token), timeout=TIMEOUT))
