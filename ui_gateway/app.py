# ui_gateway/app.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import httpx
import os
from pathlib import Path

from shared.config import SERVICE_PORTS
from shared.schemas import HealthResponse

app = FastAPI(
    title="UI Gateway",
    description="واجهة المستخدم لنظام استرجاع المعلومات",
    version="1.0.0"
)

# عناوين الخدمات
PREPROCESSING_URL = f"http://127.0.0.1:{SERVICE_PORTS['preprocessing']}"
RETRIEVAL_URL = f"http://127.0.0.1:{SERVICE_PORTS['retrieval']}"

# HTML الرئيسي
HTML_TEMPLATE = """
<!DOCTYPE html>
<html dir="ltr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔍 نظام استرجاع المعلومات</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 30px; }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .search-form { display: flex; gap: 15px; flex-wrap: wrap; }
        .search-input {
            flex: 3;
            padding: 15px 20px;
            font-size: 1.1rem;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
        }
        .search-input:focus { outline: none; border-color: #667eea; }
        .select-model, .select-k {
            padding: 15px;
            font-size: 1rem;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            background: white;
        }
        .select-model { flex: 1; }
        .select-k { flex: 0.3; }
        .search-btn {
            padding: 15px 25px;
            font-size: 1.1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .search-btn:hover { transform: translateY(-2px); }
        .results-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }
        .result-item {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
        }
        .result-item:hover { transform: translateX(5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .result-rank {
            display: inline-block;
            width: 30px;
            height: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 30px;
            font-weight: bold;
            margin-right: 15px;
        }
        .result-id { font-weight: bold; font-size: 1.1rem; color: #333; }
        .result-score { color: #28a745; font-size: 0.9rem; margin-left: 15px; }
        .tokens-info {
            background: #e3f2fd;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 0.9rem;
            color: #1565c0;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .no-results { text-align: center; color: #666; padding: 40px; }
        .footer { text-align: center; color: white; margin-top: 30px; opacity: 0.8; }
        @media (max-width: 768px) { .search-form { flex-direction: column; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 نظام استرجاع المعلومات</h1>
            <p>بحث في مجموعة Vaswani | TF-IDF, BM25, Hybrid</p>
        </div>

        <div class="card">
            <form method="post" action="/search" class="search-form">
                <input type="text" name="query" class="search-input" 
                       placeholder="اكتب سؤالك هنا..." value="{{ query }}" required>
                <select name="model" class="select-model">
                    <option value="tfidf" {{ 'selected' if model == 'tfidf' else '' }}>🔬 TF-IDF</option>
                    <option value="bm25" {{ 'selected' if model == 'bm25' else '' }}>📊 BM25</option>
                    <option value="hybrid" {{ 'selected' if model == 'hybrid' else '' }}>🔄 Hybrid</option>
                </select>
                <select name="k" class="select-k">
                    {% for i in [5,10,15,20,25,30] %}
                    <option value="{{ i }}" {{ 'selected' if k == i else '' }}>Top {{ i }}</option>
                    {% endfor %}
                </select>
                <button type="submit" class="search-btn">🔍 بحث</button>
            </form>
        </div>

        {% if error %}
        <div class="card">
            <div class="error">
                <strong>⚠️ خطأ:</strong> {{ error }}
            </div>
        </div>
        {% endif %}

        {% if query and not error %}
        <div class="card">
            <div class="results-header">
                <h2>📄 نتائج البحث عن: "{{ query }}"</h2>
                <span>{{ model.upper() }}</span>
            </div>
            
            {% if query_tokens %}
            <div class="tokens-info">
                🔍 كلمات البحث: <strong>{{ query_tokens|join(', ') }}</strong>
            </div>
            {% endif %}

            {% if results and results|length > 0 %}
                {% for result in results %}
                <div class="result-item">
                    <span class="result-rank">{{ loop.index }}</span>
                    <span class="result-id">📄 {{ result.doc_id }}</span>
                    <span class="result-score">⭐ التشابه: {{ "%.4f"|format(result.score) }}</span>
                </div>
                {% endfor %}
            {% else %}
                <div class="no-results">😕 لا توجد نتائج. حاول كلمات أخرى.</div>
            {% endif %}
        </div>
        {% endif %}

        <div class="footer">
            <p>نظام استرجاع المعلومات | SOA Architecture</p>
        </div>
    </div>
</body>
</html>
"""

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(service="ui_gateway", status="ok")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """الصفحة الرئيسية"""
    return HTMLResponse(content=HTML_TEMPLATE.replace("{{ query }}", "").replace("{{ model }}", "tfidf").replace("{{ k }}", "10"), status_code=200)

@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query: str = Form(...),
    model: str = Form("tfidf"),
    k: int = Form(10)
):
    """معالجة البحث وعرض النتائج"""
    results_data = []
    error = None
    query_tokens = []
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. معالجة الاستعلام
            process_response = await client.post(
                f"{PREPROCESSING_URL}/process-query",
                params={"query_text": query}
            )
            
            if process_response.status_code == 200:
                processed = process_response.json()
                query_tokens = processed.get("tokens", [])
                
                # 2. البحث
                search_response = await client.post(
                    f"{RETRIEVAL_URL}/search",
                    json={
                        "dataset_name": "vaswani",
                        "query_tokens": query_tokens,
                        "model_type": model,
                        "k": k
                    }
                )
                
                if search_response.status_code == 200:
                    results_data = search_response.json().get("results", [])
                else:
                    error = f"خطأ في خدمة البحث: {search_response.status_code}"
            else:
                error = f"خطأ في خدمة المعالجة: {process_response.status_code}"
                
    except httpx.ConnectError:
        error = "❌ لا يمكن الاتصال بالخدمات. تأكد من تشغيل Preprocessing (8001) و Retrieval (8003)"
    except Exception as e:
        error = f"❌ خطأ: {str(e)}"
    
    # بناء HTML النتيجة
    html = HTML_TEMPLATE
    
    # إضافة القيم
    html = html.replace("{{ query }}", query)
    html = html.replace("{{ model }}", model)
    html = html.replace("{{ k }}", str(k))
    
    # إضافة الأخطاء
    if error:
        error_html = f'<div class="card"><div class="error"><strong>⚠️ خطأ:</strong> {error}</div></div>'
        html = html.replace("{% if error %}<div class=\"card\"><div class=\"error\"><strong>⚠️ خطأ:</strong> {{ error }}</div></div>{% endif %}", error_html)
    else:
        html = html.replace("{% if error %}<div class=\"card\"><div class=\"error\"><strong>⚠️ خطأ:</strong> {{ error }}</div></div>{% endif %}", "")
    
    # إضافة النتائج
    if results_data:
        results_html = '<div class="card"><div class="results-header"><h2>📄 نتائج البحث عن: "' + query + '"</h2><span>' + model.upper() + '</span></div>'
        if query_tokens:
            results_html += f'<div class="tokens-info">🔍 كلمات البحث: <strong>{", ".join(query_tokens)}</strong></div>'
        for i, result in enumerate(results_data):
            results_html += f'''
            <div class="result-item">
                <span class="result-rank">{i+1}</span>
                <span class="result-id">📄 {result["doc_id"]}</span>
                <span class="result-score">⭐ التشابه: {result["score"]:.4f}</span>
            </div>
            '''
        results_html += '</div>'
        html = html.replace("{% if query and not error %}<div class=\"card\">...</div>{% endif %}", results_html)
    else:
        if query and not error:
            no_results_html = f'<div class="card"><div class="results-header"><h2>📄 نتائج البحث عن: "{query}"</h2><span>{model.upper()}</span></div><div class="no-results">😕 لا توجد نتائج. حاول كلمات أخرى.</div></div>'
            if query_tokens:
                no_results_html = f'<div class="card"><div class="results-header"><h2>📄 نتائج البحث عن: "{query}"</h2><span>{model.upper()}</span></div><div class="tokens-info">🔍 كلمات البحث: <strong>{", ".join(query_tokens)}</strong></div><div class="no-results">😕 لا توجد نتائج. حاول كلمات أخرى.</div></div>'
            html = html.replace("{% if query and not error %}<div class=\"card\">...</div>{% endif %}", no_results_html)
        else:
            html = html.replace("{% if query and not error %}<div class=\"card\">...</div>{% endif %}", "")
    
    return HTMLResponse(content=html, status_code=200)

@app.get("/status")
async def service_status():
    """التحقق من حالة الخدمات"""
    status = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{PREPROCESSING_URL}/health")
            status["preprocessing"] = "✅" if resp.status_code == 200 else "❌"
        except:
            status["preprocessing"] = "❌"
        
        try:
            resp = await client.get(f"{RETRIEVAL_URL}/health")
            status["retrieval"] = "✅" if resp.status_code == 200 else "❌"
        except:
            status["retrieval"] = "❌"
    
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ui_gateway.app:app", host="127.0.0.1", port=8000, reload=True)