"""
Web routes for the UI gateway.
"""
from __future__ import annotations

from fastapi import APIRouter, Cookie, Form, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from shared.config import BASE_DIR, DEFAULT_BM25_B, DEFAULT_BM25_K1, DEFAULT_DATASET_NAME, DEFAULT_IR_DATASET
from shared.schemas import HealthResponse
from ui_gateway.services.orchestrator import UIGatewayOrchestrator

router = APIRouter()
orchestrator = UIGatewayOrchestrator()

MAX_SESSION_HISTORY = 10
HISTORY_COOKIE = "ir_search_history"
HISTORY_SEP = "|||"

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
        "use_personalization": False,
        "execution_mode": "basic",
        "dataset_name": DEFAULT_DATASET_NAME,
        "ir_dataset_id": DEFAULT_IR_DATASET,
        "user_history": "",
        "fusion_mode": "rrf",
        "weight_tfidf": 0.34,
        "weight_bm25": 0.33,
        "weight_embedding": 0.33,
        "error": None,
        "results": [],
        "query_tokens": [],
        "refined_query": None,
        "personalization_applied": [],
        "suggestions": [],
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


def _parse_history_cookie(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(HISTORY_SEP) if item.strip()]


def _format_history_list(history: list[str]) -> str:
    return ", ".join(history)


def _update_history(history: list[str], query: str) -> list[str]:
    query = query.strip()
    if not query:
        return history
    updated = [query] + [item for item in history if item != query]
    return updated[:MAX_SESSION_HISTORY]


@router.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    response: Response,
    query: str = Form(...),
    model: str = Form("tfidf"),
    k: int = Form(10),
    bm25_k1: float = Form(DEFAULT_BM25_K1),
    bm25_b: float = Form(DEFAULT_BM25_B),
    use_refinement: str | None = Form(None),
    use_personalization: str | None = Form(None),
    execution_mode: str = Form("basic"),
    dataset_name: str = Form(DEFAULT_DATASET_NAME),
    user_history: str = Form(""),
    fusion_mode: str = Form("rrf"),
    weight_tfidf: float = Form(0.34),
    weight_bm25: float = Form(0.33),
    weight_embedding: float = Form(0.33),
    search_history: str | None = Cookie(default=None),
) -> HTMLResponse:
    extra_mode = execution_mode == "extra"
    refinement_enabled = extra_mode and use_refinement == "true"
    personalization_enabled = extra_mode and use_personalization == "true"

    session_history = _parse_history_cookie(search_history)
    manual_history = [h.strip() for h in user_history.split(",") if h.strip()]
    if extra_mode and personalization_enabled and manual_history:
        history_for_search = manual_history
    elif extra_mode and personalization_enabled:
        history_for_search = session_history
    else:
        history_for_search = []

    search_context = await orchestrator.handle_search(
        query=query,
        model_str=model,
        top_k=k,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        use_refinement=refinement_enabled,
        use_personalization=personalization_enabled,
        execution_mode=execution_mode,
        dataset_name=dataset_name,
        user_history=_format_history_list(history_for_search),
        fusion_mode=fusion_mode,
        weight_tfidf=weight_tfidf,
        weight_bm25=weight_bm25,
        weight_embedding=weight_embedding,
    )

    if not search_context.get("error") and query.strip():
        updated_history = _update_history(session_history, query)
        response.set_cookie(
            key=HISTORY_COOKIE,
            value=HISTORY_SEP.join(updated_history),
            max_age=86400,
            httponly=False,
            samesite="lax",
        )
        if extra_mode and personalization_enabled and not manual_history:
            search_context["user_history"] = _format_history_list(updated_history)

    return templates.TemplateResponse(
        request,
        "index.html",
        build_page_context(**search_context),
    )
