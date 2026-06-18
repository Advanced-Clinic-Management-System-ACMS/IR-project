"""
Web routes for the UI gateway.
"""
from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from shared.config import BASE_DIR, DEFAULT_BM25_B, DEFAULT_BM25_K1, DEFAULT_DATASET_NAME
from shared.schemas import HealthResponse
from ui_gateway.services.orchestrator import UIGatewayOrchestrator

router = APIRouter()
orchestrator = UIGatewayOrchestrator()

TEMPLATE_DIR = str(BASE_DIR / "ui_gateway" / "templates")
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html", "xml"]),
    cache_size=0,
)
templates = Jinja2Templates(env=jinja_env)


def build_page_context(**overrides) -> dict:
    context = {
        "query": "",
        "model": "tfidf",
        "k": 10,
        "bm25_k1": DEFAULT_BM25_K1,
        "bm25_b": DEFAULT_BM25_B,
        "use_refinement": False,
        "dataset_name": DEFAULT_DATASET_NAME,
        "error": None,
        "results": [],
        "query_tokens": [],
        "refined_query": None,
        "elapsed_ms": None,
    }
    context.update(overrides)
    return context


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(service="ui_gateway", status="ok")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        build_page_context(),
    )


@router.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query: str = Form(...),
    model: str = Form("tfidf"),
    k: int = Form(10),
    bm25_k1: float = Form(DEFAULT_BM25_K1),
    bm25_b: float = Form(DEFAULT_BM25_B),
    use_refinement: str | None = Form(None),
    dataset_name: str = Form(DEFAULT_DATASET_NAME),
) -> HTMLResponse:
    refinement_enabled = use_refinement == "true"
    search_context = await orchestrator.handle_search(
        query=query,
        model_str=model,
        top_k=k,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        use_refinement=refinement_enabled,
        dataset_name=dataset_name,
    )
    return templates.TemplateResponse(
        request,
        "index.html",
        build_page_context(**search_context),
    )
