from fastapi import Request

from ..db import Database
from ..services.registry_service import RegistryService
from ..services.run_reader import RunReader
from ..worker.supervisor import WorkerSupervisor


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_registry(request: Request) -> RegistryService:
    return request.app.state.registry


def get_reader(request: Request) -> RunReader:
    return request.app.state.reader


def get_supervisor(request: Request) -> WorkerSupervisor | None:
    return getattr(request.app.state, "supervisor", None)
