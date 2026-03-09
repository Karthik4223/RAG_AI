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
    debug=settings.DEBUG
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

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
