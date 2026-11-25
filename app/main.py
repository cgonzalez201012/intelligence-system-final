"""API REST (FastAPI) del orquestador multi-agente de Gobierno de Datos.

La API NO corre el agente dentro del request: un flujo de agentes tarda decenas de segundos y
mantener la conexion abierta tanto tiempo la cortaria algun timeout y dejaria al servidor sin recursos.
En cambio:

    POST /tasks                 valida, registra el job (PENDING) en Redis, lo encola y devuelve el job_id al instante
    GET  /tasks/{id}            consulta el estado (polling): PENDING, RUNNING, WAITING_APPROVAL, DONE, FAILED
    POST /tasks/{id}/approve    entrega la decision humana que reanuda un job pausado (ver hitl.py)
    GET  /health                comprueba que Redis responde

Todos los endpoints son `async` y solo hacen operaciones de Redis con el cliente asincrono: no
hay ninguna llamada bloqueante ni al modelo ni a la base dentro del event loop. El grafo corre en
otro proceso (app/worker.py).

Arranque:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.dependencias import almacen_de
from app.hitl import router as router_de_aprobacion
from app.jobs import AlmacenDeJobs
from app.schemas import ErrorApi, MensajeDeCola, Salud, TareaCreada, TareaEstado, TareaIn
from app.settings import Ajustes

DESCRIPCION = """
Expone el sistema de inteligencia (supervisor + investigador con RAG, analista y redactor)
como un servicio de produccion: las tareas se encolan en Redis, las corre un worker aparte y las que
tienen un efecto externo (escalar un incidente critico) se pausan hasta que una persona las apruebe.
Los datos son ficticios (Mercado Austral S.A.).
"""


@asynccontextmanager
async def _ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
    """Abre la conexion a Redis al arrancar (si no la inyectaron los tests) y la cierra al apagar."""
    if getattr(app.state, "almacen", None) is not None:
        yield
        return
    ajustes = Ajustes.desde_entorno()
    redis = Redis.from_url(ajustes.redis_url, decode_responses=True)
    app.state.almacen = AlmacenDeJobs(redis, ttl_segundos=ajustes.ttl_jobs_segundos)
    try:
        yield
    finally:
        await redis.aclose()


def crear_app(almacen: AlmacenDeJobs | None = None) -> FastAPI:
    """Arma la aplicacion. Los tests le pasan un almacen con Redis simulado."""
    app = FastAPI(
        title="API de producción: orquestador multi-agente",
        version="1.0.0",
        description=DESCRIPCION,
        lifespan=_ciclo_de_vida,
    )
    app.state.almacen = almacen
    app.include_router(router_de_aprobacion)

    @app.exception_handler(RedisError)
    async def _redis_caido(_: Request, error: RedisError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "Redis no responde: probá de nuevo en unos segundos."})

    @app.post(
        "/tasks",
        status_code=202,
        response_model=TareaCreada,
        tags=["tareas"],
        summary="Encolar una tarea",
        responses={503: {"model": ErrorApi, "description": "Redis no responde: la tarea no quedó encolada."}},
    )
    async def crear_tarea(
        cuerpo: TareaIn, respuesta: Response, almacen: AlmacenDeJobs = Depends(almacen_de)
    ) -> TareaCreada:
        """Registra el job y lo encola; responde de inmediato con el `job_id` para consultar el estado."""
        job_id = await almacen.crear(cuerpo.tarea)  # primero el registro (PENDING)...
        try:
            await almacen.encolar(MensajeDeCola(job_id=job_id, accion="ejecutar"))  # ...despues la cola
        except RedisError:
            # El job ya existe pero nunca llegara a un worker: no dejarlo PENDING para siempre.
            try:
                await almacen.actualizar(job_id, "FAILED", error="No se pudo encolar la tarea.")
            except RedisError:
                pass
            raise
        respuesta.headers["Location"] = f"/tasks/{job_id}"
        return TareaCreada(job_id=job_id, estado="PENDING", consultar_en=f"/tasks/{job_id}")

    @app.get(
        "/tasks/{job_id}",
        response_model=TareaEstado,
        tags=["tareas"],
        summary="Consultar el estado de una tarea",
        responses={
            404: {"model": ErrorApi, "description": "No existe un job con ese ID (o ya expiró)."},
            503: {"model": ErrorApi, "description": "Redis no responde."},
        },
    )
    async def estado_de_la_tarea(job_id: str, almacen: AlmacenDeJobs = Depends(almacen_de)) -> TareaEstado:
        job = await almacen.obtener(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No existe un job con ese ID (o ya expiró).")
        return job

    @app.get(
        "/health",
        response_model=Salud,
        tags=["salud"],
        summary="Estado del servicio",
        responses={503: {"model": ErrorApi, "description": "Redis no responde."}},
    )
    async def salud(almacen: AlmacenDeJobs = Depends(almacen_de)) -> Salud:
        await almacen.responde()  # si Redis no responde, el handler de RedisError devuelve 503
        return Salud(estado="ok", redis="ok", tareas_en_cola=await almacen.largo_de_la_cola())

    return app


app = crear_app()
