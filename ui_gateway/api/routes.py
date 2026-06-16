"""
This file defines the Web Routes for the UI.
It acts as the Controller, returning rendered Jinja2 HTML templates.
"""
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from shared.schemas import HealthResponse
from ui_gateway.services.orchestrator import UIGatewayOrchestrator

router = APIRouter()
orchestrator = UIGatewayOrchestrator()

# Initialize Jinja2 Templates pointing to the templates directory
templates = Jinja2Templates(directory="ui_gateway/templates")

@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(service="ui_gateway", status="ok")

@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    # Render the initial empty page
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "query": "", "model": "tfidf", "k": 10}
    )

@router.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query: str = Form(...),
    model: str = Form("tfidf"),
    k: int = Form(10)
) -> HTMLResponse:
    # Process the form data
    context = await orchestrator.handle_search(query, model, k)
    
    # Jinja2 requires the 'request' object in the context
    context["request"] = request 
    
    return templates.TemplateResponse("index.html", context)