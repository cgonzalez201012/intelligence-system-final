"""Fixtures compartidas. Ningun test necesita claves ni red."""

import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import fakeredis
import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis

from app.graph import construir_grafo_produccion
from app.jobs import AlmacenDeJobs
from app.main import crear_app
from app.notificador import NotificadorRedis
from app.schemas import MensajeDeCola
from app.settings import Ajustes
from app.worker import Servicios, procesar_mensaje
from orquestador.state import AccionPropuesta, ResultadoAccion

from orquestador.config import Configuracion
from orquestador.graph import construir_grafo
from orquestador.state import EstadoOrquestador, estado_inicial
from tests.fakes import ModeloPorPolitica, SupervisorFalso

TAREA = "Priorizá los incidentes críticos abiertos del maestro de artículos."


@pytest.fixture
def config() -> Configuracion:
    """Configuracion de prueba (sin clave real)."""
    return Configuracion(
        anthropic_api_key="clave-falsa-de-prueba",
        modelo_supervisor="modelo-falso",
        modelo_especialistas="modelo-falso",
        max_pasos=10,
        max_intentos_por_agente=3,
        recursion_especialista=12,
    )


def crear_estado(**cambios: Any) -> EstadoOrquestador:
    """Estado inicial con los campos que se quieran pisar (para tests unitarios)."""
    estado = estado_inicial(TAREA)
    estado.update(cambios)
    return estado  # type: ignore[return-value]


async def correr(
    supervisor: SupervisorFalso,
    especialistas: ModeloPorPolitica,
    *,
    max_pasos: int = 10,
    max_intentos: int = 3,
    tarea: str = TAREA,
) -> dict[str, Any]:
    """Arma el grafo con modelos falsos, lo corre de punta a punta y devuelve el estado final."""
    grafo = construir_grafo(
        supervisor, especialistas, max_pasos=max_pasos, max_intentos=max_intentos, recursion_especialista=12
    )
    resultado: dict[str, Any] = await grafo.ainvoke(
        estado_inicial(tarea), {"recursion_limit": 3 * max_pasos + 5}
    )
    return resultado


# ---------- Capa de produccion (capa de producción) ----------


@pytest.fixture
def ajustes() -> Ajustes:
    """Ajustes de prueba: sin trazas y con la concurrencia por defecto del worker."""
    return Ajustes(
        redis_url="redis://prueba",
        observabilidad=False,
        phoenix_endpoint="http://localhost:6006/v1/traces",
        phoenix_proyecto="pruebas",
        version_prompts="test",
        worker_concurrencia=5,
        ttl_jobs_segundos=3600,
        ttl_checkpoints_minutos=60,
    )


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    """Redis simulado en memoria. Con REDIS_URL_TEST apunta a un Redis real (se vacia antes de cada test)."""
    url = os.environ.get("REDIS_URL_TEST")
    if url:
        cliente = Redis.from_url(url, decode_responses=True)
        await cliente.flushdb()
    else:
        cliente = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield cliente
    await cliente.aclose()


@pytest.fixture
def almacen(redis: Redis) -> AlmacenDeJobs:
    return AlmacenDeJobs(redis, ttl_segundos=3600)


@pytest.fixture
async def cliente(almacen: AlmacenDeJobs) -> AsyncIterator[httpx.AsyncClient]:
    """Cliente HTTP que habla con la API en memoria (sin red) y con el almacen de prueba."""
    transporte = httpx.ASGITransport(app=crear_app(almacen))
    async with httpx.AsyncClient(transport=transporte, base_url="http://prueba") as http:
        yield http


class NotificadorEspia:
    """Notificador de prueba: anota cada llamada. Sirve para probar que NO se ejecuta nada sin aprobacion."""

    def __init__(self, fallar_en: set[str] | None = None) -> None:
        self.llamadas: list[tuple[str, str]] = []
        self.fallar_en = fallar_en or set()

    async def notificar(self, job_id: str, accion: AccionPropuesta) -> ResultadoAccion:
        if accion.incidente_id in self.fallar_en:
            raise ConnectionError("el servicio de avisos no responde")
        self.llamadas.append((job_id, accion.incidente_id))
        return ResultadoAccion(
            incidente_id=accion.incidente_id, destinatario=accion.destinatario, ejecutada=True, detalle="enviado"
        )


@pytest.fixture
def espia() -> NotificadorEspia:
    return NotificadorEspia()


@pytest.fixture
def armar_grafo_produccion(config: Any, espia: NotificadorEspia) -> Callable[..., CompiledStateGraph]:
    """Fabrica del grafo de produccion con modelos falsos, checkpointer en memoria y el notificador espia."""

    def armar(
        supervisor: BaseChatModel,
        especialistas: BaseChatModel,
        *,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
    ) -> CompiledStateGraph:
        return construir_grafo_produccion(
            supervisor,
            especialistas,
            config,
            checkpointer=checkpointer or InMemorySaver(),
            notificador=espia,
        )

    return armar


@pytest.fixture
def armar_servicios(
    almacen: AlmacenDeJobs, ajustes: Ajustes, config: Any, armar_grafo_produccion: Callable[..., CompiledStateGraph]
) -> Callable[..., Servicios]:
    def armar(supervisor: BaseChatModel, especialistas: BaseChatModel) -> Servicios:
        return Servicios(almacen=almacen, grafo=armar_grafo_produccion(supervisor, especialistas), ajustes=ajustes, config=config)

    return armar


async def enviar_y_procesar(servicios: Servicios, tarea: str = TAREA) -> str:
    """Hace lo mismo que la API y el worker: registra el job, lo encola, lo saca de la cola y lo procesa."""
    job_id = await servicios.almacen.crear(tarea)
    await servicios.almacen.encolar(MensajeDeCola(job_id=job_id, accion="ejecutar"))
    mensaje = await servicios.almacen.tomar_de_la_cola(0.1)
    assert mensaje is not None
    await procesar_mensaje(mensaje, servicios)
    return job_id


async def reanudar_y_procesar(servicios: Servicios) -> None:
    """Saca de la cola el mensaje de reanudacion (el que encola /approve) y lo procesa."""
    mensaje = await servicios.almacen.tomar_de_la_cola(0.1)
    assert mensaje is not None and mensaje.accion == "reanudar"
    await procesar_mensaje(mensaje, servicios)
