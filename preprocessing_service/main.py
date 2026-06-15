# preprocessing_service/main.py
from fastapi import FastAPI, HTTPException

from preprocessing_service.processor import TextPreprocessor
from shared.schemas import (
    DocumentInput,
    HealthResponse,
    PreprocessRequest,
    PreprocessResponse,
    ProcessedDocument,
)

app = FastAPI(
    title="Preprocessing Service",
    description="Text cleaning, tokenization, and stemming using NLTK.",
    version="1.0.0",
)

preprocessor = TextPreprocessor()


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        service="preprocessing_service",
        status="ok",
        details={"stemmer": "PorterStemmer", "library": "NLTK"},
    )


@app.post("/process", response_model=ProcessedDocument)
def process_single_document(document: DocumentInput) -> ProcessedDocument:
    """معالجة وثيقة واحدة"""
    try:
        return preprocessor.process_document(document)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-batch", response_model=PreprocessResponse)
def process_batch(request: PreprocessRequest) -> PreprocessResponse:
    """معالجة مجموعة من الوثائق"""
    try:
        processed = [preprocessor.process_document(doc) for doc in request.documents]
        return PreprocessResponse(processed=processed, count=len(processed))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-query", response_model=dict)
def process_query(query_text: str) -> dict:
    """معالجة استعلام بحث (نص واحد) وإرجاع tokens"""
    try:
        # إنشاء DocumentInput مؤقت للمعالجة
        temp_doc = DocumentInput(doc_id="query", text=query_text)
        processed = preprocessor.process_document(temp_doc)
        return {
            "query_id": "temp",
            "original_text": query_text,
            "tokens": processed.tokens,
            "token_count": len(processed.tokens)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    from shared.config import SERVICE_PORTS

    uvicorn.run(
        "preprocessing_service.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS["preprocessing"],
        reload=True,
    )