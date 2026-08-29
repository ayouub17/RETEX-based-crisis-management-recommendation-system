"""FastAPI web interface for the RETEX recommendation system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from main import find_similar_retex, get_dashboard_stats
from recommendations.engine import RecommendationEngine


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="RETEX Crisis Recommender",
    description="Recherche sémantique et recommandations pour le management des crises.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class CrisisRequest(BaseModel):
    """Payload sent by the web form or another client application."""

    description: str = Field(min_length=10, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=20)
    limit: int = Field(default=10, ge=1, le=30)


class RecommendationResponse(BaseModel):
    """Response containing source RETEX and derived action proposals."""

    similar_cases: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    stats: dict[str, Any] | None = None


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the single-page user interface."""
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Provide a lightweight endpoint for server availability checks."""
    return {"status": "ok"}


@app.get("/api/dashboard")
def dashboard_stats() -> dict[str, Any]:
    """Expose the current RETEX corpus metrics used by the dashboard."""
    try:
        return get_dashboard_stats()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/recommendations", response_model=RecommendationResponse)
def get_recommendations(payload: CrisisRequest) -> dict[str, Any]:
    """Find similar RETEX and aggregate their operational proposals."""
    try:
        similar_cases = find_similar_retex(payload.description, payload.top_k)
        if not similar_cases:
            similar_cases = []
        proposals = RecommendationEngine().recommend(similar_cases, payload.limit)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {"similar_cases": similar_cases, "stats": get_dashboard_stats(), **proposals}
