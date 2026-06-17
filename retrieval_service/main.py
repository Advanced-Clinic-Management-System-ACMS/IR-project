from fastapi import FastAPI
from retrieval_service.api.routes import router

app = FastAPI(title="IR Retrieval System")

app.include_router(router)