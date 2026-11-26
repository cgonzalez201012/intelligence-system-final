"""Casos borde de la revisión final: cola corrupta, TTL vencido, aprobaciones ambiguas, citas con otro formato."""

import logging
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from redis.asyncio import Redis

from app.jobs import AlmacenDeJobs
from app.schemas import MensajeDeCola
from app.settings import Ajustes
from app.worker import describir_falla
from orquestador.errores import ErrorConfiguracion
from orquestador.state import DecisionHumana, FragmentoCitado
from orquestador.validacion import codigos_citados, ids_citados, validar_redactor
from tests.fakes import analisis_de_datos, hallazgos_de_datos


# ---------- cola corrupta ----------


@pytest.mark.parametrize("basura", ["esto-no-es-json", '{"job_id": 123}', "{}", ""])
async def test_un_mensaje_corrupto_se_descarta_y_no_tumba_al_worker(
    almacen: AlmacenDeJobs, redis: Redis, basura: str, caplog: pytest.LogCaptureFixture
) -> None:
    await redis.lpush("cola:tareas", basura)
    with caplog.at_level(logging.ERROR):
        assert await almacen.tomar_de_la_cola(espera_segundos=0.1) is None
    assert "Descarto un mensaje inválido" in caplog.text
    # el siguiente mensaje bueno sí se atiende
    await almacen.encolar(MensajeDeCola(job_id="a" * 32, accion="ejecutar"))
    mensaje = await almacen.tomar_de_la_cola(espera_segundos=0.1)
    assert mensaje is not None and mensaje.job_id == "a" * 32


# ---------- registro expirado ----------


async def test_actualizar_no_recrea_un_job_que_ya_expiro(almacen: AlmacenDeJobs, redis: Redis) -> None:
    job_id = await almacen.crear("Priorizá los incidentes críticos abiertos")
    await redis.delete(f"job:{job_id}")
    await almacen.actualizar(job_id, "DONE")
    assert await redis.exists(f"job:{job_id}") == 0
    assert await almacen.obtener(job_id) is None


async def test_un_registro_incompleto_se_trata_como_inexistente(almacen: AlmacenDeJobs, redis: Redis) -> None:
    job_id = "b" * 32
    await redis.hset(f"job:{job_id}", mapping={"estado": "DONE"})
    assert await almacen.obtener(job_id) is None


def test_los_ttl_incoherentes_se_rechazan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TTL_JOBS_SEGUNDOS", "60")
    monkeypatch.setenv("JOB_TIMEOUT_SEGUNDOS", "300")
    with pytest.raises(ErrorConfiguracion, match="TTL_JOBS_SEGUNDOS"):
        Ajustes.desde_entorno(cargar_dotenv=False)
    monkeypatch.setenv("TTL_JOBS_SEGUNDOS", "86400")
    monkeypatch.setenv("TTL_CHECKPOINTS_MINUTOS", "10")
    with pytest.raises(ErrorConfiguracion, match="TTL_CHECKPOINTS_MINUTOS"):
        Ajustes.desde_entorno(cargar_dotenv=False)


# ---------- aprobación humana ----------


@pytest.mark.parametrize("aprobada", ["yes", 1, "true", "1", None])
def test_la_aprobacion_exige_un_booleano_de_verdad(aprobada: Any) -> None:
    with pytest.raises(ValidationError):
        DecisionHumana(aprobada=aprobada, revisor="Cris")


@pytest.mark.parametrize("revisor", ["   ", "\x00\x07", "Cris\x1b[2J"])
def test_el_revisor_no_puede_estar_vacio_ni_tener_caracteres_de_control(revisor: str) -> None:
    with pytest.raises(ValidationError):
        DecisionHumana(aprobada=True, revisor=revisor)


def test_el_revisor_se_guarda_sin_espacios_de_mas() -> None:
    assert DecisionHumana(aprobada=True, revisor="  Cris  ").revisor == "Cris"


async def test_la_api_rechaza_aprobaciones_ambiguas_y_tareas_con_caracteres_de_control(
    cliente: httpx.AsyncClient, almacen: AlmacenDeJobs
) -> None:
    job_id = await almacen.crear("Priorizá los incidentes críticos abiertos")
    await almacen.actualizar(job_id, "WAITING_APPROVAL")
    for cuerpo in ({"aprobada": "yes", "revisor": "Cris"}, {"aprobada": True, "revisor": "   "}):
        assert (await cliente.post(f"/tasks/{job_id}/approve", json=cuerpo)).status_code == 422
    assert (await cliente.post("/tasks", json={"tarea": "Priorizá incidentes \x00 críticos"})).status_code == 422


# ---------- citas con otro formato ----------


@pytest.mark.parametrize(
    "texto",
    ["gd-pol-001", "GD POL 001", "GD‑POL‑001", "GD-POL-001-v2", "GD-POL-0010"],
)
def test_los_codigos_de_politica_se_reconocen_con_cualquier_formato(texto: str) -> None:
    assert len(codigos_citados(f"Según {texto} el plazo es de 4 h.")) == 1


def test_los_ids_se_reconocen_sin_importar_mayusculas_ni_cantidad_de_digitos() -> None:
    assert ids_citados("inc-1001 y INC-99999") == {"INC-1001", "INC-99999"}


@pytest.mark.parametrize("cita", ["gd-pol-009", "GD POL 009", "GD-POL-0091"])
def test_citar_una_politica_que_no_se_recupero_se_detecta_aunque_cambie_el_formato(cita: str) -> None:
    hallazgos = hallazgos_de_datos().model_copy(
        update={
            "politicas": [
                FragmentoCitado(
                    chunk_id="c1", documento_id="d1", codigo="GD-POL-001", fuente="01.md", pagina=1,
                    cita="GD-POL-001 (01.md, pág. 1)", texto="texto",
                )
            ]
        }
    )
    problemas = validar_redactor(hallazgos, analisis_de_datos(hallazgos), "INC-1001 INC-1002. " + f"Según {cita}, el plazo es de 4 h.")
    assert any("no se recuperaron" in p for p in problemas)


# ---------- mensajes de falla ----------


def test_un_429_se_explica_y_no_muestra_el_cuerpo_del_proveedor() -> None:
    class Limite(Exception):
        status_code = 429

    texto = describir_falla(Limite("Error code: 429 - {'type': 'rate_limit_error'}"))
    assert "429" in texto and "rate_limit_error" not in texto


def test_describir_falla_oculta_claves() -> None:
    texto = describir_falla(RuntimeError("falló con " + "sk-" + "ant-api03-SECRETO123 y " + "pcsk" + "_SECRETO456 y " + "AIza" + "SECRETO789"))
    assert "SECRETO" not in texto and "[clave oculta]" in texto
