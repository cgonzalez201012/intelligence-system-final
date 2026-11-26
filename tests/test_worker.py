"""El worker: corre el grafo fuera de la API, deja el resultado en Redis y NUNCA deja un job colgado."""

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

import httpx
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from redis.asyncio import Redis

from app.graph import construir_grafo_produccion
from app.jobs import AlmacenDeJobs
from app.notificador import BANDEJA_DE_SALIDA, NotificadorRedis
from app.schemas import MensajeDeCola
from app.settings import Ajustes
from app.worker import Servicios, correr_worker, describir_falla, procesar_mensaje
from orquestador.config import Configuracion
from orquestador.state import EstadoOrquestador
from tests.conftest import TAREA, enviar_y_procesar, reanudar_y_procesar
from tests.fakes import (
    ErrorDeApi,
    SupervisorFalso,
    decidir,
    especialistas_falsos,
    guion_analista,
    guion_investigador,
    informe,
)

APROBACION = {"aprobada": True, "revisor": "Cris", "comentario": "ok"}


def _supervisor() -> SupervisorFalso:
    return SupervisorFalso(
        guion=[decidir("investigador", "Recolectá."), decidir("analista", "Calculá."), decidir("redactor", "Escribí."), decidir("FINISH")]
    )


def _criticos() -> Any:
    return especialistas_falsos(guion_investigador(), guion_analista(), [informe("INC-1001", "INC-1002")])


def _sin_criticos() -> Any:
    return especialistas_falsos(guion_investigador(("INC-1003",), severidad="alta"), guion_analista(("INC-1003",)), [informe("INC-1003")])


class Medidor:
    """Cuenta cuantos jobs estan corriendo A LA VEZ dentro del grafo."""

    def __init__(self) -> None:
        self.activos = 0
        self.maximo = 0
        self.corridas = 0


def _grafo_de_un_nodo(nodo: Callable[..., Any]) -> Any:
    grafo = StateGraph(EstadoOrquestador)
    grafo.add_node("unico", nodo)
    grafo.add_edge(START, "unico")
    grafo.add_edge("unico", END)
    return grafo.compile(checkpointer=InMemorySaver())


def _servicios_con(grafo: Any, almacen: AlmacenDeJobs, ajustes: Ajustes, config: Configuracion) -> Servicios:
    return Servicios(almacen=almacen, grafo=grafo, ajustes=ajustes, config=config)


def _grafo_lento(medidor: Medidor, segundos: float) -> Any:
    async def nodo(estado: EstadoOrquestador) -> dict[str, Any]:
        medidor.corridas += 1
        medidor.activos += 1
        medidor.maximo = max(medidor.maximo, medidor.activos)
        await asyncio.sleep(segundos)  # async de verdad: mientras uno espera, los demas avanzan
        medidor.activos -= 1
        return {"estado_final": "completado", "respuesta_final": "listo", "pasos": 1}

    return _grafo_de_un_nodo(nodo)


# ---------- Camino feliz ----------


async def test_un_job_sin_incidentes_criticos_termina_en_done_con_su_resultado(
    armar_servicios: Callable[..., Servicios], almacen: AlmacenDeJobs
) -> None:
    servicios = armar_servicios(_supervisor(), _sin_criticos())
    job_id = await enviar_y_procesar(servicios)
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "DONE"
    assert job.resultado is not None
    assert job.resultado.estado_final == "completado"
    assert "INC-1003" in job.resultado.respuesta
    assert job.resultado.agentes_que_intervinieron == ["investigador", "analista", "redactor"]
    assert job.resultado.delegaciones == 3
    assert job.resultado.acciones.propuestas == [] and job.resultado.acciones.decision is None
    assert job.finalizado_en is not None and job.error is None
    assert job.uso.llamadas_al_modelo > 0  # el contador de uso se engancho a los callbacks del grafo


async def test_un_job_critico_se_pausa_en_waiting_approval_con_lo_que_hay_que_aprobar(
    armar_servicios: Callable[..., Servicios], almacen: AlmacenDeJobs, espia: Any
) -> None:
    servicios = armar_servicios(_supervisor(), _criticos())
    job_id = await enviar_y_procesar(servicios)
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "WAITING_APPROVAL"
    assert job.finalizado_en is None and job.resultado is None
    assert job.aprobacion_pendiente is not None
    assert [a.incidente_id for a in job.aprobacion_pendiente.acciones] == ["INC-1001", "INC-1002"]
    assert espia.llamadas == []  # pausado: todavia no se hizo nada


async def test_flujo_completo_pausa_aprobacion_por_la_api_y_reanudacion(
    armar_servicios: Callable[..., Servicios], almacen: AlmacenDeJobs, cliente: httpx.AsyncClient, espia: Any
) -> None:
    servicios = armar_servicios(_supervisor(), _criticos())
    job_id = await enviar_y_procesar(servicios)
    antes = (await almacen.obtener(job_id))
    assert antes is not None and antes.estado == "WAITING_APPROVAL"

    assert (await cliente.post(f"/tasks/{job_id}/approve", json=APROBACION)).status_code == 202
    await reanudar_y_procesar(servicios)

    job = (await cliente.get(f"/tasks/{job_id}")).json()
    assert job["estado"] == "DONE"
    acciones = job["resultado"]["acciones"]
    assert acciones["decision"]["revisor"] == "Cris" and acciones["decision"]["aprobada"] is True
    assert [e["incidente_id"] for e in acciones["ejecutadas"]] == ["INC-1001", "INC-1002"]
    assert espia.llamadas == [(job_id, "INC-1001"), (job_id, "INC-1002")]
    assert job["uso"]["llamadas_al_modelo"] >= antes.uso.llamadas_al_modelo  # el uso se acumula entre las dos etapas
    assert job["segundos_de_ejecucion"] >= antes.segundos_de_ejecucion
    assert job["aprobacion_pendiente"] is None


async def test_si_la_persona_rechaza_el_job_termina_sin_ejecutar_nada(
    armar_servicios: Callable[..., Servicios], almacen: AlmacenDeJobs, cliente: httpx.AsyncClient, espia: Any
) -> None:
    servicios = armar_servicios(_supervisor(), _criticos())
    job_id = await enviar_y_procesar(servicios)
    await cliente.post(f"/tasks/{job_id}/approve", json={"aprobada": False, "revisor": "Cris", "comentario": "esperemos"})
    await reanudar_y_procesar(servicios)
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "DONE" and job.resultado is not None
    assert job.resultado.acciones.decision is not None and job.resultado.acciones.decision.aprobada is False
    assert job.resultado.acciones.ejecutadas == []
    assert job.resultado.respuesta  # igual se entrega el informe
    assert espia.llamadas == []


async def test_con_el_notificador_real_el_escalamiento_queda_en_la_bandeja_de_redis(
    redis: Redis, almacen: AlmacenDeJobs, ajustes: Ajustes, config: Configuracion, cliente: httpx.AsyncClient
) -> None:
    grafo = construir_grafo_produccion(
        _supervisor(), _criticos(), config, checkpointer=InMemorySaver(), notificador=NotificadorRedis(redis, ttl_segundos=60)
    )
    servicios = _servicios_con(grafo, almacen, ajustes, config)
    job_id = await enviar_y_procesar(servicios)
    assert await redis.llen(BANDEJA_DE_SALIDA) == 0  # type: ignore[misc]  # pausado: nada salio todavia
    await cliente.post(f"/tasks/{job_id}/approve", json=APROBACION)
    await reanudar_y_procesar(servicios)
    avisos = [json.loads(a) for a in await redis.lrange(BANDEJA_DE_SALIDA, 0, -1)]  # type: ignore[misc]
    assert sorted(a["incidente_id"] for a in avisos) == ["INC-1001", "INC-1002"]
    assert {a["destinatario"] for a in avisos} == {"Gerente Comercial (Data Owner de Articulos)"}


# ---------- Fallas: el job siempre termina en FAILED, nunca queda colgado ----------


async def test_una_excepcion_dentro_del_grafo_deja_el_job_en_failed_y_no_escapa_del_worker(
    almacen: AlmacenDeJobs, ajustes: Ajustes, config: Configuracion
) -> None:
    async def nodo_roto(estado: EstadoOrquestador) -> dict[str, Any]:
        raise RuntimeError("se rompió algo adentro\nsegunda línea que no debe verse")

    servicios = _servicios_con(_grafo_de_un_nodo(nodo_roto), almacen, ajustes, config)
    job_id = await enviar_y_procesar(servicios)  # si la excepcion escapara, este await fallaria
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "FAILED"
    assert job.error == "RuntimeError: se rompió algo adentro"
    assert job.finalizado_en is not None and job.resultado is None


async def test_una_clave_invalida_del_proveedor_deja_el_job_en_failed_con_un_motivo_claro(
    armar_servicios: Callable[..., Servicios], almacen: AlmacenDeJobs
) -> None:
    servicios = armar_servicios(_supervisor(), especialistas_falsos(investigador=[ErrorDeApi("invalid x-api-key", 401)]))
    job_id = await enviar_y_procesar(servicios)
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "FAILED"
    assert "credenciales" in (job.error or "") and "ANTHROPIC_API_KEY" in (job.error or "")


async def test_si_el_grafo_supera_su_limite_de_pasos_el_job_termina_en_failed(
    almacen: AlmacenDeJobs, ajustes: Ajustes, config: Configuracion
) -> None:
    async def nodo_que_se_pasa(estado: EstadoOrquestador) -> dict[str, Any]:
        raise GraphRecursionError("limite")

    servicios = _servicios_con(_grafo_de_un_nodo(nodo_que_se_pasa), almacen, ajustes, config)
    job_id = await enviar_y_procesar(servicios)
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "FAILED" and "límite de pasos" in (job.error or "")


async def test_un_job_que_falla_no_frena_a_los_siguientes(
    almacen: AlmacenDeJobs, ajustes: Ajustes, config: Configuracion
) -> None:
    corridas = 0

    async def a_veces_roto(estado: EstadoOrquestador) -> dict[str, Any]:
        nonlocal corridas
        corridas += 1
        if corridas == 1:
            raise RuntimeError("primero falla")
        return {"estado_final": "completado", "respuesta_final": "bien", "pasos": 1}

    servicios = _servicios_con(_grafo_de_un_nodo(a_veces_roto), almacen, ajustes, config)
    primero = await enviar_y_procesar(servicios)
    segundo = await enviar_y_procesar(servicios)
    assert (await almacen.obtener(primero)).estado == "FAILED"  # type: ignore[union-attr]
    assert (await almacen.obtener(segundo)).estado == "DONE"  # type: ignore[union-attr]


def test_describir_falla_es_corto_seguro_y_no_filtra_trazas() -> None:
    assert "credenciales" in describir_falla(ErrorDeApi("x", 401))
    assert "credenciales" in describir_falla(ErrorDeApi("x", 403))
    assert "nombre del modelo" in describir_falla(ErrorDeApi("x", 404))
    assert describir_falla(GraphRecursionError("x")).startswith("El grafo superó")
    assert describir_falla(ValueError("")) == "ValueError"
    assert describir_falla(ValueError("a" * 500)) == "ValueError: " + "a" * 200
    assert "\n" not in describir_falla(ValueError("linea1\nlinea2"))


# ---------- Mensajes raros ----------


async def test_un_mensaje_repetido_no_corre_el_job_dos_veces(almacen: AlmacenDeJobs, ajustes: Ajustes, config: Configuracion) -> None:
    medidor = Medidor()
    servicios = _servicios_con(_grafo_lento(medidor, 0), almacen, ajustes, config)
    job_id = await enviar_y_procesar(servicios)
    await procesar_mensaje(MensajeDeCola(job_id=job_id, accion="ejecutar"), servicios)  # llega de nuevo
    assert medidor.corridas == 1  # el job ya no estaba PENDING: se ignora


async def test_un_mensaje_de_un_job_inexistente_se_ignora(almacen: AlmacenDeJobs, ajustes: Ajustes, config: Configuracion) -> None:
    medidor = Medidor()
    servicios = _servicios_con(_grafo_lento(medidor, 0), almacen, ajustes, config)
    await procesar_mensaje(MensajeDeCola(job_id="fantasma", accion="ejecutar"), servicios)
    assert medidor.corridas == 0


async def test_reanudar_sin_decision_registrada_deja_el_job_en_failed(almacen: AlmacenDeJobs, ajustes: Ajustes, config: Configuracion) -> None:
    servicios = _servicios_con(_grafo_lento(Medidor(), 0), almacen, ajustes, config)
    job_id = await almacen.crear(TAREA)
    await procesar_mensaje(MensajeDeCola(job_id=job_id, accion="reanudar"), servicios)
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "FAILED" and "decisión" in (job.error or "")


# ---------- Concurrencia (el motivo de usar una cola y un worker asincrono) ----------


async def _esperar_terminales(almacen: AlmacenDeJobs, ids: list[str], limite: float = 10) -> None:
    fin = time.monotonic() + limite
    while time.monotonic() < fin:
        trabajos = [await almacen.obtener(i) for i in ids]
        if all(t is not None and t.estado in ("DONE", "FAILED") for t in trabajos):
            return
        await asyncio.sleep(0.02)
    raise AssertionError("los jobs no terminaron a tiempo")


async def test_cinco_jobs_concurrentes_corren_a_la_vez_y_no_uno_tras_otro(
    almacen: AlmacenDeJobs, ajustes: Ajustes, config: Configuracion, cliente: httpx.AsyncClient
) -> None:
    medidor = Medidor()
    servicios = _servicios_con(_grafo_lento(medidor, 0.4), almacen, ajustes, config)
    detener = asyncio.Event()
    worker = asyncio.create_task(correr_worker(servicios, detener))
    inicio = time.perf_counter()
    respuestas = await asyncio.gather(*[cliente.post("/tasks", json={"tarea": f"{TAREA} ({i})"}) for i in range(5)])
    ids = [r.json()["job_id"] for r in respuestas]
    await _esperar_terminales(almacen, ids)
    total = time.perf_counter() - inicio
    detener.set()
    await worker
    estados = [(await almacen.obtener(i)).estado for i in ids]  # type: ignore[union-attr]
    assert estados == ["DONE"] * 5
    assert medidor.maximo == 5  # los cinco estuvieron dentro del grafo al mismo tiempo
    assert total < 1.2  # en serie serian 5 x 0.4 = 2 s


async def test_el_worker_no_corre_mas_jobs_a_la_vez_que_su_concurrencia(
    almacen: AlmacenDeJobs, ajustes: Ajustes, config: Configuracion, cliente: httpx.AsyncClient
) -> None:
    medidor = Medidor()
    limitado = Ajustes(**{**ajustes.__dict__, "worker_concurrencia": 2})
    servicios = _servicios_con(_grafo_lento(medidor, 0.15), almacen, limitado, config)
    detener = asyncio.Event()
    worker = asyncio.create_task(correr_worker(servicios, detener))
    ids = [(await cliente.post("/tasks", json={"tarea": f"{TAREA} ({i})"})).json()["job_id"] for i in range(6)]
    await _esperar_terminales(almacen, ids)
    detener.set()
    await worker
    assert medidor.corridas == 6 and medidor.maximo == 2


async def test_la_api_responde_mientras_el_worker_esta_ocupado(
    almacen: AlmacenDeJobs, ajustes: Ajustes, config: Configuracion, cliente: httpx.AsyncClient
) -> None:
    servicios = _servicios_con(_grafo_lento(Medidor(), 0.6), almacen, ajustes, config)
    detener = asyncio.Event()
    worker = asyncio.create_task(correr_worker(servicios, detener))
    job_id = (await cliente.post("/tasks", json={"tarea": TAREA})).json()["job_id"]
    await asyncio.sleep(0.15)  # el job esta corriendo
    inicio = time.perf_counter()
    consulta = await cliente.get(f"/tasks/{job_id}")
    assert time.perf_counter() - inicio < 0.3  # la consulta no espera al job
    assert consulta.json()["estado"] == "RUNNING"
    await _esperar_terminales(almacen, [job_id])
    detener.set()
    await worker


async def test_al_detenerlo_el_worker_deja_terminar_los_jobs_en_curso(
    almacen: AlmacenDeJobs, ajustes: Ajustes, config: Configuracion, cliente: httpx.AsyncClient
) -> None:
    servicios = _servicios_con(_grafo_lento(Medidor(), 0.3), almacen, ajustes, config)
    detener = asyncio.Event()
    worker = asyncio.create_task(correr_worker(servicios, detener))
    job_id = (await cliente.post("/tasks", json={"tarea": TAREA})).json()["job_id"]
    await asyncio.sleep(0.1)
    detener.set()  # SIGTERM en medio del job
    await asyncio.wait_for(worker, timeout=5)
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "DONE"


def test_describir_falla_distingue_la_falta_de_saldo_de_un_modelo_mal_escrito() -> None:
    sin_saldo = describir_falla(ErrorDeApi("Error code: 400 - Your credit balance is too low to access the Anthropic API.", 400))
    assert "saldo" in sin_saldo and "nombre del modelo" not in sin_saldo
    assert "nombre del modelo" in describir_falla(ErrorDeApi("model: claude-xyz not found", 400))
