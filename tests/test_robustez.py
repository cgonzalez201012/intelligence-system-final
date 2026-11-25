"""Casos borde: jobs huérfanos, tiempo máximo, IDs inválidos, respuestas raras de la API y ajustes del Compose."""

import asyncio
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
import yaml
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from redis.asyncio import Redis

from app.jobs import AlmacenDeJobs
from app.schemas import MensajeDeCola
from app.settings import Ajustes
from app.worker import MOTIVO_HUERFANO, MOTIVO_TIMEOUT, Servicios, barrer_huerfanos, correr_worker, procesar_mensaje
from orquestador.config import BASE_DIR, Configuracion
from orquestador.errores import ErrorConfiguracion
from orquestador.state import EstadoOrquestador
from scripts.carga import correr_peticion
from tests.conftest import TAREA

HACE_UNA_HORA = (datetime.now(UTC) - timedelta(hours=1)).isoformat(timespec="milliseconds")


async def _job_en(almacen: AlmacenDeJobs, redis: Redis, estado: str, *, actualizado_en: str = HACE_UNA_HORA) -> str:
    job_id = await almacen.crear(TAREA)
    await redis.hset(f"job:{job_id}", mapping={"estado": estado, "actualizado_en": actualizado_en})
    return job_id


# ---------- Jobs huérfanos (el worker murió a mitad de un job) ----------


async def test_un_job_viejo_en_running_pasa_a_failed(almacen: AlmacenDeJobs, redis: Redis) -> None:
    job_id = await _job_en(almacen, redis, "RUNNING")
    marcados = await almacen.marcar_huerfanos(mas_viejos_que_segundos=360, motivo=MOTIVO_HUERFANO)
    job = await almacen.obtener(job_id)
    assert marcados == [job_id]
    assert job is not None and job.estado == "FAILED" and job.error == MOTIVO_HUERFANO and job.finalizado_en is not None


async def test_el_barrido_no_toca_los_jobs_que_estan_vivos_o_que_no_corren(almacen: AlmacenDeJobs, redis: Redis) -> None:
    ahora = datetime.now(UTC).isoformat(timespec="milliseconds")
    reciente = await _job_en(almacen, redis, "RUNNING", actualizado_en=ahora)
    otros = [await _job_en(almacen, redis, estado) for estado in ("PENDING", "WAITING_APPROVAL", "DONE", "FAILED")]
    await redis.set(f"job:{reciente}:reclamo", "1")  # clave auxiliar: no es un job
    assert await almacen.marcar_huerfanos(mas_viejos_que_segundos=360, motivo=MOTIVO_HUERFANO) == []
    estados = [(await almacen.obtener(i)).estado for i in [reciente, *otros]]  # type: ignore[union-attr]
    assert estados == ["RUNNING", "PENDING", "WAITING_APPROVAL", "DONE", "FAILED"]


async def test_el_barrido_ignora_un_registro_sin_fecha_o_con_fecha_ilegible(almacen: AlmacenDeJobs, redis: Redis) -> None:
    ilegible = await _job_en(almacen, redis, "RUNNING", actualizado_en="ayer")
    sin_fecha = await almacen.crear(TAREA)
    await redis.hset(f"job:{sin_fecha}", mapping={"estado": "RUNNING"})
    await redis.hdel(f"job:{sin_fecha}", "actualizado_en")
    assert await almacen.marcar_huerfanos(mas_viejos_que_segundos=1, motivo="x") == []
    assert (await almacen.obtener(ilegible)).estado == "RUNNING"  # type: ignore[union-attr]


async def test_el_worker_barre_los_huerfanos_mientras_trabaja(
    almacen: AlmacenDeJobs, redis: Redis, ajustes: Ajustes
) -> None:
    job_id = await _job_en(almacen, redis, "RUNNING")
    servicios = Servicios(almacen=almacen, grafo=None, ajustes=ajustes, config=None)  # type: ignore[arg-type]
    detener = asyncio.Event()
    tarea = asyncio.create_task(barrer_huerfanos(servicios, detener, 0.05))
    await asyncio.sleep(0.2)
    detener.set()
    await asyncio.wait_for(tarea, timeout=2)
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "FAILED" and job.error == MOTIVO_HUERFANO


async def test_correr_worker_arranca_el_barrido_y_lo_cierra_al_detenerse(
    almacen: AlmacenDeJobs, redis: Redis, ajustes: Ajustes, config: Configuracion
) -> None:
    job_id = await _job_en(almacen, redis, "RUNNING")
    servicios = Servicios(almacen=almacen, grafo=None, ajustes=ajustes, config=config)  # type: ignore[arg-type]
    detener = asyncio.Event()
    worker = asyncio.create_task(correr_worker(servicios, detener, barrido_cada_segundos=0.05))
    await asyncio.sleep(0.3)
    detener.set()
    await asyncio.wait_for(worker, timeout=5)  # si el barrido quedara colgado, esto no terminaria
    assert (await almacen.obtener(job_id)).estado == "FAILED"  # type: ignore[union-attr]


# ---------- Tiempo máximo de un job ----------


async def test_un_job_que_supera_el_tiempo_maximo_queda_en_failed(
    almacen: AlmacenDeJobs, ajustes: Ajustes, config: Configuracion
) -> None:
    async def nodo_que_no_termina(estado: EstadoOrquestador) -> dict[str, Any]:
        await asyncio.sleep(30)
        return {}

    grafo = StateGraph(EstadoOrquestador)
    grafo.add_node("unico", nodo_que_no_termina)
    grafo.add_edge(START, "unico")
    grafo.add_edge("unico", END)
    corto = Ajustes(**{**ajustes.__dict__, "job_timeout_segundos": 0.2})
    servicios = Servicios(almacen=almacen, grafo=grafo.compile(checkpointer=InMemorySaver()), ajustes=corto, config=config)
    job_id = await almacen.crear(TAREA)
    await procesar_mensaje(MensajeDeCola(job_id=job_id, accion="ejecutar"), servicios)
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "FAILED" and job.error == MOTIVO_TIMEOUT


def test_el_tiempo_maximo_se_configura_por_entorno_y_se_valida(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JOB_TIMEOUT_SEGUNDOS", raising=False)
    por_defecto = Ajustes.desde_entorno(cargar_dotenv=False)
    assert por_defecto.job_timeout_segundos == 300 and por_defecto.job_huerfano_segundos > por_defecto.job_timeout_segundos
    monkeypatch.setenv("JOB_TIMEOUT_SEGUNDOS", "90")
    assert Ajustes.desde_entorno(cargar_dotenv=False).job_timeout_segundos == 90
    monkeypatch.setenv("JOB_TIMEOUT_SEGUNDOS", "5")
    with pytest.raises(ErrorConfiguracion):
        Ajustes.desde_entorno(cargar_dotenv=False)


# ---------- IDs que no son de un job ----------


@pytest.mark.parametrize("identificador", ["no-existe", "ABC", "x" * 300, "0" * 32 + ":reclamo", "%2e%2e"])
async def test_un_id_con_formato_invalido_es_404_y_nunca_503(
    cliente: httpx.AsyncClient, almacen: AlmacenDeJobs, redis: Redis, identificador: str
) -> None:
    await redis.set(f"job:{'0' * 32}:reclamo", "1")  # clave de otro tipo: hgetall sobre ella daria WRONGTYPE
    assert (await cliente.get(f"/tasks/{identificador}")).status_code == 404
    aprobacion = await cliente.post(f"/tasks/{identificador}/approve", json={"aprobada": True, "revisor": "Cris"})
    assert aprobacion.status_code == 404


# ---------- Prueba de carga: respuestas inesperadas ----------


async def test_si_la_api_devuelve_un_error_al_consultar_el_script_lo_informa_y_no_se_cae() -> None:
    def api_rota(pedido: httpx.Request) -> httpx.Response:
        if pedido.method == "POST":
            return httpx.Response(202, json={"job_id": "a" * 32})
        return httpx.Response(503, text="Redis no responde")

    async with httpx.AsyncClient(transport=httpx.MockTransport(api_rota), base_url="http://x") as cliente:
        resultado = await correr_peticion(cliente, 1, TAREA, aprobar=False, tiempo_maximo=5, cada=0.01)
    assert resultado.estado_final != "DONE" and "503" in (resultado.error or "")


async def test_si_la_api_responde_algo_que_no_es_json_el_script_lo_informa() -> None:
    def api_rara(pedido: httpx.Request) -> httpx.Response:
        if pedido.method == "POST":
            return httpx.Response(202, json={"job_id": "a" * 32})
        return httpx.Response(200, text="<html>no soy json</html>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(api_rara), base_url="http://x") as cliente:
        resultado = await correr_peticion(cliente, 1, TAREA, aprobar=False, tiempo_maximo=5, cada=0.01)
    assert resultado.estado_final == "FAILED" and resultado.error


# ---------- Compose: lo que se configura en .env tiene que llegar a los contenedores ----------


def test_toda_variable_del_env_example_llega_al_worker_en_docker_compose() -> None:
    nombres = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", (BASE_DIR / ".env.example").read_text(encoding="utf-8"), re.M))
    compose = yaml.safe_load((BASE_DIR / "docker-compose.yml").read_text(encoding="utf-8"))
    entorno_del_worker = set(compose["services"]["worker"]["environment"])
    # TTL_JOBS_SEGUNDOS la leen la API y el worker; REDIS_URL y las demas tambien estan en el entorno del worker.
    assert nombres - entorno_del_worker == set(), "estas variables se documentan pero no llegan al contenedor"
    assert "TTL_JOBS_SEGUNDOS" in compose["services"]["api"]["environment"]


def test_el_worker_tiene_tiempo_para_terminar_sus_jobs_al_apagarse() -> None:
    """Un job puede durar hasta JOB_TIMEOUT_SEGUNDOS: si Docker corta antes, el job queda a medias."""
    from app.settings import JOB_TIMEOUT_POR_DEFECTO

    compose = yaml.safe_load((BASE_DIR / "docker-compose.yml").read_text(encoding="utf-8"))
    gracia = compose["services"]["worker"]["stop_grace_period"]
    segundos = int(gracia[:-1]) * {"s": 1, "m": 60}[gracia[-1]]
    assert segundos > JOB_TIMEOUT_POR_DEFECTO, f"{gracia} no alcanza para un job de {JOB_TIMEOUT_POR_DEFECTO} s"


def test_el_diagrama_tambien_esta_como_imagen() -> None:
    assert (BASE_DIR / "docs" / "grafo.png").stat().st_size > 5_000
