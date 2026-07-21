"""FastAPI application factory and SPA host."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .db import Database
from .routers import catalog, jobs, model_tests, runs, templates
from .services.registry_service import RegistryService
from .services.run_reader import RunReader
from .worker.supervisor import WorkerSupervisor


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.db_path)
    db.initialize()
    app.state.db = db
    app.state.registry = RegistryService(settings, db)
    app.state.reader = RunReader()
    supervisor = None
    if os.getenv("FACTOR_WEBAPP_DISABLE_WORKER", "").lower() not in {"1", "true", "yes"}:
        supervisor = WorkerSupervisor(db, settings)
        supervisor.start()
    app.state.supervisor = supervisor
    try:
        yield
    finally:
        if supervisor:
            supervisor.stop()


def create_app() -> FastAPI:
    application = FastAPI(
        title="因子评价平台",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/api/health", tags=["system"])
    def health() -> dict:
        return {"status": "ok"}

    application.include_router(catalog.router, prefix="/api")
    application.include_router(templates.router, prefix="/api")
    application.include_router(jobs.router, prefix="/api")
    application.include_router(model_tests.router, prefix="/api")
    application.include_router(runs.router, prefix="/api")

    dist = settings.frontend_dist
    if (dist / "assets").is_dir():
        application.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @application.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(404)
        index = dist / "index.html"
        if not index.is_file():
            raise HTTPException(
                404, "Frontend is not built. Run npm install && npm run build in webapp/frontend."
            )
        return FileResponse(index)

    return application


app = create_app()
