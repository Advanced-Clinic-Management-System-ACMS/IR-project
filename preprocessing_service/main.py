from fastapi import FastAPI

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
    return preprocessor.process_document(document)


@app.post("/process-batch", response_model=PreprocessResponse)
def process_batch(request: PreprocessRequest) -> PreprocessResponse:
    processed = [preprocessor.process_document(doc) for doc in request.documents]
    return PreprocessResponse(processed=processed, count=len(processed))


if __name__ == "__main__":
    import uvicorn

    from shared.config import SERVICE_PORTS

    uvicorn.run(
        "preprocessing_service.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS["preprocessing"],
        reload=True,
    )
