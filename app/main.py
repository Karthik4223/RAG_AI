from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.api.routes import router
from app.core.config import settings
from app.core.logging import setup_logging
import uvicorn

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    debug=settings.DEBUG,
    version="1.0.0",
    description="A production-ready RAG system powered by Google Gemini 2.5 Flash.",
    contact={
        "name": "Karthik",
        "url": "https://github.com/Karthik4223/RAG_AI",
    }
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
setup_logging()

# Include routes
app.include_router(router, prefix=settings.API_V1_STR)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/visualize", response_class=HTMLResponse)
async def get_visualizer():
    with open("app/static/visualizer.html", "r") as f:
        return f.read()

@app.get("/visualize-query", response_class=HTMLResponse)
async def get_query_visualizer():
    with open("app/static/query_visualizer.html", "r") as f:
        return f.read()

@app.get("/architecture", response_class=HTMLResponse)
async def get_architecture():
    with open("app/static/architecture.html", "r") as f:
        return f.read()

@app.get("/comparison", response_class=HTMLResponse)
async def get_comparison():
    with open("app/static/comparison.html", "r") as f:
        return f.read()

@app.get("/implementation", response_class=HTMLResponse)
async def get_implementation():
    with open("app/static/implementation_flow.html", "r") as f:
        return f.read()

@app.get("/chat", response_class=HTMLResponse)
async def get_chat():
    with open("app/static/chat.html", "r") as f:
        return f.read()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>RAG AI Ecosystem</title>
            <style>
                body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; background: #030712; color: #f8fafc; }
                .card { background: #0f172a; padding: 2rem; border-radius: 1rem; border: 1px solid #1e293b; text-align: center; }
                a { color: #6366f1; text-decoration: none; display: block; margin: 0.5rem 0; font-weight: bold; }
                a:hover { text-decoration: underline; }
                h1 { color: #6366f1; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>Explore RAG</h1>
                <p>Welcome to the production-ready RAG system.</p>
                <a href="/chat">🚀 Live Semantic Chat</a>
                <a href="/implementation">🎯 Implementation Strategy (New)</a>
                <a href="/architecture">🏗️ Architecture Visualization</a>
                <a href="/comparison">📊 LangChain vs LlamaIndex</a>
                <a href="/visualize">🧪 Ingestion Trace</a>
                <a href="/visualize-query">🔍 Query Trace</a>
                <a href="/docs">📜 API Documentation</a>
            </div>
        </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
