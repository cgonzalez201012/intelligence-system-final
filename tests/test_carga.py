"""El script de la prueba de carga: sus calculos (p50/p95) y su recorrido completo contra la API."""

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.jobs import AlmacenDeJobs
from app.main import crear_app
from app.settings import Ajustes
from app.worker import Servicios, correr_worker
from orquestador.config import Configuracion
from scripts.carga import (
    TAREAS_POR_PERFIL,
    ResultadoJob,
    correr_peticion,
    correr_prueba,
    costo_usd,
    parsear,
    percentil,
    resumir,
)
from tests.test_worker import Medidor, _criticos, _grafo_lento, _servicios_con, _sin_criticos, _supervisor


def test_el_percentil_usa_el_metodo_del_rango_mas_cercano() -> None:
    muestras = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentil(muestras, 50) == 3.0
    assert percentil(muestras, 95) == 5.0  # con 5 datos, el p95 es el maximo
    assert percentil([7.0], 95) == 7.0
    assert percentil([], 95) is None
    cien = [float(i) for i in range(1, 101)]
    assert (percentil(cien, 50), percentil(cien, 95), percentil(cien, 99)) == (50.0, 95.0, 99.0)


def test_el_costo_necesita_los_precios_por_millon_de_tokens() -> None:
    assert costo_usd(1_000_000, 1_000_000, 1.0, 5.0) == pytest.approx(6.0)
    assert costo_usd(10_000, 2_000, 1.0, 5.0) == pytest.approx(0.02)
    assert costo_usd(10_000, 2_000, None, 5.0) is None


def _job(i: int, estado: str, total: float, entrada: int = 0, salida: int = 0) -> ResultadoJob:
    return ResultadoJob(indice=i, job_id=f"j{i}", estado_final=estado, segundos_totales=total, tokens_entrada=entrada, tokens_salida=salida)


def test_el_resumen_solo_cuenta_la_latencia_de_los_jobs_exitosos() -> None:
    resultados = [_job(1, "DONE", 10, 1000, 200), _job(2, "DONE", 20, 3000, 400), _job(3, "FAILED", 99)]
    resumen = resumir(
        resultados, url="u", perfil="p", inicio="t", segundos_de_pared=21.0, precio_entrada=1.0, precio_salida=5.0
    )
    assert (resumen.exitosas, resumen.fallidas) == (2, 1)
    assert (resumen.latencia_p50_s, resumen.latencia_p95_s, resumen.latencia_maxima_s) == (10.0, 20.0, 20.0)
    assert resumen.tokens_por_ejecucion_promedio == 2300.0
    assert resumen.costo_por_ejecucion_usd == pytest.approx((4000 * 1 + 600 * 5) / 1e6 / 2)


def test_sin_precios_no_se_inventa_un_costo() -> None:
    resumen = resumir([_job(1, "DONE", 5, 100, 10)], url="u", perfil="p", inicio="t", segundos_de_pared=5.0)
    assert resumen.costo_por_ejecucion_usd is None


def test_cada_perfil_tiene_cinco_tareas_distintas() -> None:
    for tareas in TAREAS_POR_PERFIL.values():
        assert len(tareas) == len(set(tareas)) == 5


def test_los_argumentos_tienen_valores_por_defecto_razonables() -> None:
    args = parsear([])
    assert (args.n, args.perfil, args.url) == (5, "sin-aprobacion", "http://localhost:8000")


async def _con_worker(servicios: Servicios, trabajo: Callable[[], Any]) -> Any:
    detener = asyncio.Event()
    worker = asyncio.create_task(correr_worker(servicios, detener))
    try:
        return await trabajo()
    finally:
        detener.set()
        await worker


async def test_una_peticion_sin_criticos_termina_y_queda_medida(
    armar_servicios: Callable[..., Servicios], cliente: httpx.AsyncClient
) -> None:
    servicios = armar_servicios(_supervisor(), _sin_criticos())
    resultado = await _con_worker(
        servicios,
        lambda: correr_peticion(cliente, 1, TAREAS_POR_PERFIL["sin-aprobacion"][0], aprobar=False, tiempo_maximo=10, cada=0.02),
    )
    assert resultado.estado_final == "DONE" and resultado.error is None
    assert 0 < resultado.segundos_hasta_aceptado < resultado.segundos_totales


async def test_con_aprobacion_el_script_aprueba_y_el_job_termina(
    armar_servicios: Callable[..., Servicios], cliente: httpx.AsyncClient
) -> None:
    servicios = armar_servicios(_supervisor(), _criticos())
    resultado = await _con_worker(
        servicios,
        lambda: correr_peticion(cliente, 1, TAREAS_POR_PERFIL["con-aprobacion"][0], aprobar=True, tiempo_maximo=10, cada=0.02),
    )
    assert resultado.estado_final == "DONE" and resultado.aprobado_por_el_script


async def test_sin_aprobar_el_job_se_informa_como_pausado_y_no_como_exitoso(
    armar_servicios: Callable[..., Servicios], cliente: httpx.AsyncClient
) -> None:
    servicios = armar_servicios(_supervisor(), _criticos())
    resultado = await _con_worker(
        servicios,
        lambda: correr_peticion(cliente, 1, TAREAS_POR_PERFIL["con-aprobacion"][0], aprobar=False, tiempo_maximo=10, cada=0.02),
    )
    assert resultado.estado_final == "WAITING_APPROVAL" and not resultado.aprobado_por_el_script


async def test_la_prueba_dispara_las_cinco_a_la_vez_y_el_worker_las_corre_en_paralelo(
    almacen: AlmacenDeJobs, ajustes: Ajustes, config: Configuracion
) -> None:
    medidor = Medidor()
    servicios = _servicios_con(_grafo_lento(medidor, 0.3), almacen, ajustes, config)
    transporte = httpx.ASGITransport(app=crear_app(almacen))
    resultados, segundos = await _con_worker(
        servicios,
        lambda: correr_prueba(
            "http://prueba", TAREAS_POR_PERFIL["sin-aprobacion"], aprobar=False, tiempo_maximo=10, cada=0.02,
            transporte=transporte,
        ),
    )
    assert [r.estado_final for r in resultados] == ["DONE"] * 5
    assert medidor.maximo == 5
    assert segundos < 1.2  # en serie serian 5 x 0.3 = 1.5 s
    assert all(r.segundos_hasta_aceptado < 0.3 for r in resultados)  # la API acepta sin esperar al grafo
