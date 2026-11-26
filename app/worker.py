"""Worker: saca tareas de la cola de Redis, corre el grafo y deja el resultado (o el error) en Redis.

Es un proceso aparte de la API (`python -m app.worker`) y se puede escalar con mas instancias:
como la cola, el estado de los jobs y los checkpoints estan en Redis, ninguna instancia depende de las otras.

Ciclo de vida de un job:   PENDING -> RUNNING -> DONE
                                          |-> WAITING_APPROVAL -> (POST /approve) -> PENDING -> RUNNING -> DONE
                                          `-> FAILED          (cualquier excepcion del grafo)

Una excepcion dentro del grafo NUNCA puede quedar sin capturar: si pasara, el job quedaria en RUNNING
para siempre y el cliente haria polling sin fin. Por eso `procesar_mensaje` la convierte siempre en FAILED.
Dos redes de seguridad mas contra el mismo problema:

- cada job tiene un tiempo maximo (`JOB_TIMEOUT_SEGUNDOS`): si lo supera, queda en FAILED;
- si el worker muere a mitad de un job (kill, falta de memoria, reinicio), ningun codigo de este proceso
  puede marcarlo, asi que cada worker barre periodicamente los jobs que llevan en RUNNING mas que el
  timeout. Si era una REANUDACION (ya tenia la decision humana y su checkpoint), se reencola para que
  otro worker la retome (`rescatar_reanudaciones_huerfanas`); si no, pasa a FAILED (`marcar_huerfanos`).
"""

import asyncio
import re
import logging
import signal
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.graph import abrir_checkpointer, crear_grafo_con_modelos_reales, verificar_modulos_de_redis
from app.jobs import AlmacenDeJobs
from app.notificador import NotificadorRedis
from app.observability import etiquetas_de_traza, iniciar_observabilidad
from app.schemas import AccionesDelJob, AprobacionPendiente, MensajeDeCola, ResultadoTarea, TareaEstado, UsoDelModelo
from app.settings import Ajustes
from orquestador.config import Configuracion
from orquestador.errores import ErrorConfiguracion
from orquestador.state import FragmentoCitado, estado_inicial
from orquestador.trazas import ContadorDeUso

logger = logging.getLogger("worker")

# El grafo de produccion tiene 3 nodos mas que el del modulo 6 (proponer, aprobar, ejecutar).
MARGEN_DE_RECURSION = 6

INTERVALO_DE_BARRIDO_SEGUNDOS = 30
MOTIVO_TIMEOUT = "La tarea superó el tiempo máximo de ejecución y se canceló."
MOTIVO_HUERFANO = "El worker que corría esta tarea se detuvo antes de terminarla. Volvé a enviarla."


@dataclass
class Servicios:
    """Todo lo que necesita el worker para trabajar (los tests lo arman con modelos falsos)."""

    almacen: AlmacenDeJobs
    grafo: CompiledStateGraph
    ajustes: Ajustes
    config: Configuracion


# ---------- Un job ----------


# Claves de Anthropic, Pinecone y Google: si aparecen en un mensaje de error, no se devuelven al cliente.
PATRON_SECRETOS = re.compile(r"(?:sk-ant-|pcsk_|AIza)[\w\-]+")


def describir_falla(error: Exception) -> str:
    """Texto corto y seguro (sin trazas ni datos internos) de por que fallo un job."""
    estado = getattr(error, "status_code", None)
    if isinstance(error, GraphRecursionError):
        return "El grafo superó su límite de pasos sin cerrar la tarea."
    if estado in (401, 403):
        return f"El proveedor del modelo rechazó las credenciales (HTTP {estado}). Revisá ANTHROPIC_API_KEY."
    if "credit balance" in str(error).lower():  # Anthropic responde 400 cuando se acaba el saldo
        return "La cuenta de Anthropic no tiene saldo suficiente (HTTP 400). Cargá crédito en Plans & Billing y volvé a enviar la tarea."
    if estado in (400, 404):
        return f"El proveedor del modelo rechazó la solicitud (HTTP {estado}). Revisá el nombre del modelo."
    if estado == 429:
        return "El proveedor del modelo limitó el uso (HTTP 429). Esperá unos minutos y volvé a enviar la tarea."
    if isinstance(estado, int) and estado >= 500:
        return f"El proveedor del modelo está saturado o caído (HTTP {estado}). Esperá unos minutos y volvé a enviar la tarea."
    if type(error).__name__ in ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout"):
        return "No pude conectarme con el proveedor del modelo. Revisá la conexión y volvé a enviar la tarea."
    detalle = str(error).strip().splitlines()[0][:200] if str(error).strip() else ""
    detalle = PATRON_SECRETOS.sub("[clave oculta]", detalle)
    return f"{type(error).__name__}{': ' + detalle if detalle else ''}"


def _config_de_ejecucion(job_id: str, contador: ContadorDeUso, servicios: Servicios) -> RunnableConfig:
    return {
        "configurable": {"thread_id": job_id},  # un hilo del checkpointer por job
        "recursion_limit": servicios.config.recursion_limit_grafo + MARGEN_DE_RECURSION,
        "callbacks": [contador],
        "run_name": "tarea",
        "tags": [f"job:{job_id}"],
        "metadata": {"job_id": job_id},
    }


def fuentes_citadas(politicas: list[FragmentoCitado], respuesta: str) -> list[str]:
    """Politicas del RAG que la respuesta realmente nombra (trazabilidad de cada afirmacion).

    Si la respuesta copia la cita completa ("GD-POL-001 (01-politica.md, pág. 1)") se lista esa pagina. Si solo
    nombra el codigo, se lista el documento ("GD-POL-001 (01-politica.md)") sin inventar una pagina que no se cito.
    """
    citadas: set[str] = set()
    for fragmento in politicas:
        if fragmento.cita in respuesta:
            citadas.add(fragmento.cita)
        elif re.search(rf"(?<![\w-]){re.escape(fragmento.codigo)}(?![\w-])", respuesta):
            citadas.add(f"{fragmento.codigo} ({fragmento.fuente})")
    # una cita exacta ya cubre al mismo documento nombrado solo por codigo
    exactas = {c.partition(" (")[0] for c in citadas if ", pág." in c}
    return sorted(c for c in citadas if ", pág." in c or c.partition(" (")[0] not in exactas)


def resultado_desde_estado(valores: dict[str, Any]) -> ResultadoTarea:
    """Arma el resultado de la API a partir del estado final del grafo."""
    agentes: list[str] = []
    for contribucion in valores.get("contribuciones", []):
        if contribucion.agente not in agentes:
            agentes.append(contribucion.agente)
    respuesta = valores.get("respuesta_final") or ""
    hallazgos = valores.get("hallazgos")
    fuentes = fuentes_citadas(hallazgos.politicas if hallazgos is not None else [], respuesta)
    return ResultadoTarea(
        estado_final=valores.get("estado_final") or "desconocido",
        respuesta=respuesta,
        fuentes_citadas=fuentes,
        agentes_que_intervinieron=agentes,
        delegaciones=int(valores.get("pasos", 0)),
        acciones=AccionesDelJob(
            propuestas=list(valores.get("acciones_propuestas", [])),
            decision=valores.get("aprobacion"),
            ejecutadas=list(valores.get("acciones_ejecutadas", [])),
        ),
    )


def _uso_acumulado(job: TareaEstado, contador: ContadorDeUso) -> dict[str, int]:
    """Uso del modelo del job: lo que ya tenia (corridas anteriores, antes de una pausa) mas lo de esta corrida."""
    return job.uso.sumar(
        UsoDelModelo(
            llamadas_al_modelo=contador.llamadas,
            tokens_entrada=contador.tokens_entrada,
            tokens_salida=contador.tokens_salida,
        )
    ).model_dump()


async def procesar_mensaje(mensaje: MensajeDeCola, servicios: Servicios) -> None:
    """Corre (o reanuda) un job. Nunca deja escapar una excepcion del grafo: la convierte en FAILED."""
    almacen = servicios.almacen
    job = await almacen.obtener(mensaje.job_id)
    if job is None or job.estado != "PENDING":
        logger.warning("Mensaje ignorado: el job %s no existe o no está PENDING.", mensaje.job_id)
        return
    if mensaje.accion == "reanudar" and job.decision is None:
        await almacen.actualizar(job.job_id, "FAILED", error="Se pidió reanudar un job sin decisión humana registrada.")
        return
    if mensaje.accion == "reanudar":
        # Si el checkpoint expiró (o Redis se vació), reanudar correría el grafo desde cero y fallaría con un KeyError.
        foto = await servicios.grafo.aget_state({"configurable": {"thread_id": job.job_id}})
        if not foto.next:
            await almacen.actualizar(
                job.job_id, "FAILED", error="El estado pausado de esta tarea expiró. Volvé a enviarla."
            )
            return

    await almacen.actualizar(job.job_id, "RUNNING")
    logger.info("job %s: RUNNING (%s)", job.job_id, mensaje.accion)
    contador = ContadorDeUso()
    inicio = time.perf_counter()
    limite = asyncio.timeout(servicios.ajustes.job_timeout_segundos)
    try:
        config = _config_de_ejecucion(job.job_id, contador, servicios)
        entrada: Any
        if mensaje.accion == "ejecutar":
            entrada = estado_inicial(job.tarea)
        else:
            assert job.decision is not None  # verificado arriba
            entrada = Command(resume=job.decision.model_dump(mode="json"))
        async with limite:
            with etiquetas_de_traza(job.job_id, servicios.ajustes.version_prompts, mensaje.accion):
                await servicios.grafo.ainvoke(entrada, config)
        await _publicar(job, contador, time.perf_counter() - inicio, servicios, config)
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001 - ver docstring del modulo: toda falla termina en FAILED
        logger.exception("job %s: falló.", job.job_id)
        motivo = MOTIVO_TIMEOUT if limite.expired() else describir_falla(error)
        await _marcar_fallido(job, motivo, contador, time.perf_counter() - inicio, almacen)


async def _publicar(
    job: TareaEstado, contador: ContadorDeUso, segundos: float, servicios: Servicios, config: RunnableConfig
) -> None:
    """Despues de correr el grafo: o quedo pausado esperando aprobacion, o termino."""
    uso = _uso_acumulado(job, contador)
    total = round(job.segundos_de_ejecucion + segundos, 3)
    foto = await servicios.grafo.aget_state(config)
    if foto.next:  # el grafo se detuvo antes de terminar: esta esperando una decision
        if not foto.interrupts:
            raise RuntimeError("El grafo quedó detenido sin pedir aprobación.")
        await servicios.almacen.actualizar(
            job.job_id,
            "WAITING_APPROVAL",
            aprobacion_pendiente=AprobacionPendiente.model_validate(foto.interrupts[0].value),
            uso=uso,
            segundos_de_ejecucion=total,
        )
        logger.info("job %s: WAITING_APPROVAL", job.job_id)
        return
    resultado = resultado_desde_estado(foto.values)
    await servicios.almacen.actualizar(
        job.job_id,
        "DONE",
        resultado=resultado.model_dump(mode="json"),
        uso=uso,
        segundos_de_ejecucion=total,
    )
    logger.info("job %s: DONE (%s)", job.job_id, resultado.estado_final)


async def _marcar_fallido(
    job: TareaEstado, motivo: str, contador: ContadorDeUso, segundos: float, almacen: AlmacenDeJobs
) -> None:
    uso = _uso_acumulado(job, contador)
    try:
        await almacen.actualizar(
            job.job_id,
            "FAILED",
            error=motivo,
            aprobacion_pendiente=None,
            uso=uso,
            segundos_de_ejecucion=round(job.segundos_de_ejecucion + segundos, 3),
        )
    except RedisError:
        logger.exception("job %s: no pude registrar el FAILED (Redis no responde).", job.job_id)


# ---------- Bucle ----------


async def _procesar_y_liberar(mensaje: MensajeDeCola, servicios: Servicios, cupos: asyncio.Semaphore) -> None:
    try:
        await procesar_mensaje(mensaje, servicios)
    except Exception:  # noqa: BLE001 - una falla en un job no puede tumbar el bucle del worker
        logger.exception("job %s: error inesperado al procesar el mensaje.", mensaje.job_id)
    finally:
        cupos.release()


async def barrer_huerfanos(servicios: Servicios, detener: asyncio.Event, cada_segundos: float) -> None:
    """Cada tanto, pasa a FAILED los jobs que quedaron en RUNNING porque su worker murio (ver el docstring del modulo)."""
    while not detener.is_set():
        try:
            rescatados = await servicios.almacen.rescatar_reanudaciones_huerfanas(
                mas_viejos_que_segundos=servicios.ajustes.job_huerfano_segundos
            )
            if rescatados:
                logger.warning("Reanudaciones interrumpidas devueltas a la cola: %s", ", ".join(rescatados))
            huerfanos = await servicios.almacen.marcar_huerfanos(
                mas_viejos_que_segundos=servicios.ajustes.job_huerfano_segundos, motivo=MOTIVO_HUERFANO
            )
            if huerfanos:
                logger.warning("Jobs huérfanos pasados a FAILED: %s", ", ".join(huerfanos))
        except RedisError:
            logger.exception("No pude barrer los jobs huérfanos (Redis no responde): reintento más tarde.")
        try:
            await asyncio.wait_for(detener.wait(), timeout=cada_segundos)
        except TimeoutError:
            pass


async def correr_worker(
    servicios: Servicios, detener: asyncio.Event, *, barrido_cada_segundos: float = INTERVALO_DE_BARRIDO_SEGUNDOS
) -> None:
    """Procesa mensajes hasta que `detener` se active; hasta N jobs a la vez, cada uno como tarea asincrona."""
    cupos = asyncio.Semaphore(servicios.ajustes.worker_concurrencia)
    en_curso: set[asyncio.Task[None]] = set()
    barrido = asyncio.create_task(barrer_huerfanos(servicios, detener, barrido_cada_segundos))
    try:
        logger.info("Worker listo (concurrencia %d).", servicios.ajustes.worker_concurrencia)
        while not detener.is_set():
            await cupos.acquire()  # no sacar de la cola mas de lo que se puede atender
            try:
                mensaje = await servicios.almacen.tomar_de_la_cola(espera_segundos=1)
            except RedisError:
                logger.exception("No pude leer la cola de Redis: reintento en 2 s.")
                cupos.release()
                await asyncio.sleep(2)
                continue
            except Exception:  # noqa: BLE001 - nada que venga de la cola puede tumbar el bucle
                logger.exception("Error inesperado al tomar un mensaje de la cola: sigo con el siguiente.")
                cupos.release()
                continue
            if mensaje is None:
                cupos.release()
                continue
            tarea = asyncio.create_task(_procesar_y_liberar(mensaje, servicios, cupos))
            en_curso.add(tarea)
            tarea.add_done_callback(en_curso.discard)
        if en_curso:
            logger.info("Esperando que terminen %d job(s) en curso...", len(en_curso))
            await asyncio.gather(*en_curso, return_exceptions=True)
    finally:
        barrido.cancel()
        with suppress(asyncio.CancelledError):
            await barrido


# ---------- Arranque ----------


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ajustes = Ajustes.desde_entorno()
    config = Configuracion.desde_entorno()
    iniciar_observabilidad(ajustes)

    redis = Redis.from_url(ajustes.redis_url, decode_responses=True)
    try:
        await verificar_modulos_de_redis(redis)
        async with abrir_checkpointer(ajustes.redis_url, ttl_minutos=ajustes.ttl_checkpoints_minutos) as checkpointer:
            notificador = NotificadorRedis(redis, ttl_segundos=ajustes.ttl_jobs_segundos)
            grafo = crear_grafo_con_modelos_reales(config, checkpointer=checkpointer, notificador=notificador)
            servicios = Servicios(
                almacen=AlmacenDeJobs(redis, ttl_segundos=ajustes.ttl_jobs_segundos),
                grafo=grafo,
                ajustes=ajustes,
                config=config,
            )
            detener = asyncio.Event()
            bucle = asyncio.get_running_loop()
            for senal in (signal.SIGINT, signal.SIGTERM):
                bucle.add_signal_handler(senal, detener.set)
            await correr_worker(servicios, detener)
    finally:
        await redis.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ErrorConfiguracion as error:
        logger.error("Error de configuración: %s", error)
        raise SystemExit(2) from error
