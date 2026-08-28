.PHONY: install api worker frontend dev test compose down fmt

install:
	pip install -r requirements.txt

# Run the FastAPI backend (dev, auto-reload). Ingestion runs via BackgroundTasks.
api:
	uvicorn app.main:app --reload --port 8000

# Run the standalone async ingestion worker (production-style).
worker:
	python -m worker.ingestion_worker

# Run the Streamlit UI.
frontend:
	BACKEND_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py

# One-liner dev: API in background + frontend in foreground.
dev:
	@echo "Start 'make api' in one terminal and 'make frontend' in another."

test:
	pytest -q

# Full production-like stack (Qdrant + Postgres + api + worker + frontend).
compose:
	docker compose up --build

down:
	docker compose down -v
