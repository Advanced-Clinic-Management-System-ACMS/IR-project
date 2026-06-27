"""
This is the application entry point.
It initializes the FastAPI framework, registers routers, and configures the web server.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from preprocessing_service.api.routes import router

app = FastAPI(
    title="Preprocessing Service",
    description="Text cleaning, tokenization, and stemming using NLTK.",
    version="1.0.0",
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    from shared.config import SERVICE_PORTS

    uvicorn.run(
        "preprocessing_service.main:app",
        host="127.0.0.1",
        port=SERVICE_PORTS["preprocessing"],
        reload=True,
    )

# Path(__file__).resolve() → تحصل على المسار الكامل للملف الحالي

# .parent.parent → تنتقل مجلدين  (إلى مجلد المشروع الرئيسي)

# sys.path.insert(0, ...) → تضيف هذا المسار إلى قائمة المسارات التي يبحث فيها Python 













    #لتششغيل الخدمة نقطة الدخول 