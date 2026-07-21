from fastapi import Request

from ..db import Database
from ..services.registry_service import RegistryService
from ..services.run_reader import RunReader
from ..worker.supervisor import WorkerSupervisor
from ..worker.genetic_supervisor import GeneticMiningSupervisor


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_registry(request: Request) -> RegistryService:
    return request.app.state.registry


def get_reader(request: Request) -> RunReader:
    return request.app.state.reader


def get_supervisor(request: Request) -> WorkerSupervisor | None:
    return getattr(request.app.state, "supervisor", None)


def get_genetic_supervisor(request: Request) -> GeneticMiningSupervisor | None:
    return getattr(request.app.state, "genetic_supervisor", None)
