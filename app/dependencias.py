"""Dependencias de FastAPI: de donde saca cada endpoint el acceso a Redis."""

from fastapi import Request

from app.jobs import AlmacenDeJobs


def almacen_de(request: Request) -> AlmacenDeJobs:
    """El almacen de jobs que se creo al arrancar la aplicacion (los tests pueden reemplazarlo)."""
    almacen: AlmacenDeJobs = request.app.state.almacen
    return almacen
