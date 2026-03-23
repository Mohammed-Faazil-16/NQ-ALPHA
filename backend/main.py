from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router as api_router
from backend.api.user_routes import router as user_router
from backend.config import settings
from backend.db.postgres import create_tables
from backend.services.runtime_warmup import start_runtime_warmup


@asynccontextmanager
async def lifespan(_app: FastAPI):
    create_tables()
    start_runtime_warmup()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
app.include_router(user_router)


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "service": settings.app_name}
