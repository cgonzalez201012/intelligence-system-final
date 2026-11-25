"""El script de evidencia: corre cada caso contra la API (aquí simulada) y juzga si pasó lo esperado."""

import json

import httpx
import pytest

from scripts.corridas_evidencia import CASOS, Caso, Corrida, correr_caso, evaluar, tabla_markdown


def _job(estado: str, **extra) -> dict:
    base = {"job_id": "a" * 32, "estado": estado, "uso": {"tokens_entrada": 10, "tokens_salida": 5}, "segundos_de_ejecucion": 1.5}
    return {**base, **extra}


def _resultado(final="completado", respuesta="ok", fuentes=(), ejecutadas=0) -> dict:
    return {
        "estado_final": final, "respuesta": respuesta, "agentes_que_intervinieron": ["investigador", "redactor"],
        "fuentes_citadas": list(fuentes), "delegaciones": 3,
        "acciones": {"propuestas": [], "decision": None, "ejecutadas": [{"ejecutada": True}] * ejecutadas},
    }


def _api(secuencia: list[dict], aprobaciones: list[dict]) -> httpx.AsyncClient:
    estados = iter(secuencia)

    def manejar(peticion: httpx.Request) -> httpx.Response:
        if peticion.method == "POST" and peticion.url.path == "/tasks":
            return httpx.Response(202, json={"job_id": "a" * 32, "estado": "PENDING", "consultar_en": "/x"})
        if peticion.url.path.endswith("/approve"):
            aprobaciones.append(json.loads(peticion.content))
            return httpx.Response(202, json={"job_id": "a" * 32, "estado": "PENDING", "consultar_en": "/x"})
        return httpx.Response(200, json=next(estados))

    return httpx.AsyncClient(transport=httpx.MockTransport(manejar), base_url="http://prueba")


def test_hay_al_menos_cinco_casos_y_cubren_aprobar_rechazar_rag_y_fuera_de_alcance() -> None:
    assert len(CASOS) >= 5
    assert {c.decision for c in CASOS} >= {"aprobar", "rechazar", None}
    assert any(c.espera_fuentes for c in CASOS) and any(c.espera_fuera_de_alcance for c in CASOS)


async def test_un_caso_de_politicas_que_cita_fuentes_queda_ok() -> None:
    caso = Caso("p", "¿Cuál es el plazo para resolver un incidente?", espera_fuentes=True)
    cliente = _api([_job("RUNNING"), _job("DONE", resultado=_resultado(fuentes=["GD-POL-001 (01-politica.md, pág. 1)"]))], [])
    async with cliente:
        c = await correr_caso(cliente, caso, timeout_s=5)
    assert c.veredicto == "OK" and c.fuentes_citadas == ["GD-POL-001 (01-politica.md, pág. 1)"] and c.tokens_entrada == 10


async def test_un_caso_pausado_se_aprueba_o_rechaza_segun_lo_previsto() -> None:
    aprobaciones: list[dict] = []
    caso = Caso("a", "Priorizá los incidentes críticos de artículos", decision="aprobar", espera_pausa=True)
    cliente = _api([_job("WAITING_APPROVAL"), _job("DONE", resultado=_resultado(ejecutadas=2))], aprobaciones)
    async with cliente:
        c = await correr_caso(cliente, caso, timeout_s=5)
    assert aprobaciones[0]["aprobada"] is True and aprobaciones[0]["revisor"]
    assert c.veredicto == "OK" and c.paso_por_aprobacion and c.acciones_ejecutadas == 2

    aprobaciones.clear()
    caso = Caso("r", "Priorizá los incidentes críticos de proveedores", decision="rechazar", espera_pausa=True)
    cliente = _api([_job("WAITING_APPROVAL"), _job("DONE", resultado=_resultado())], aprobaciones)
    async with cliente:
        c = await correr_caso(cliente, caso, timeout_s=5)
    assert aprobaciones[0]["aprobada"] is False and c.veredicto == "OK" and c.acciones_ejecutadas == 0


@pytest.mark.parametrize(
    ("caso", "corrida", "fragmento"),
    [
        (Caso("x", "t", espera_fuentes=True), Corrida("x", "t", estado="DONE"), "citara políticas"),
        (Caso("x", "t"), Corrida("x", "t", estado="FAILED", error="boom"), "FAILED"),
        (Caso("x", "t", decision="aprobar", espera_pausa=True), Corrida("x", "t", estado="DONE", paso_por_aprobacion=True), "ninguna acción"),
        (Caso("x", "t", decision="rechazar", espera_pausa=True), Corrida("x", "t", estado="DONE", paso_por_aprobacion=True, acciones_ejecutadas=1), "se ejecutaron"),
        (Caso("x", "t", espera_pausa=True), Corrida("x", "t", estado="DONE"), "pausa"),
        (Caso("x", "t", espera_fuera_de_alcance=True), Corrida("x", "t", estado="DONE", respuesta="La capital es París."), "fuera de alcance"),
    ],
)
def test_evaluar_marca_revisar_cuando_no_pasa_lo_esperado(caso: Caso, corrida: Corrida, fragmento: str) -> None:
    veredicto, motivos = evaluar(caso, corrida)
    assert veredicto == "REVISAR" and any(fragmento in m for m in motivos)


async def test_si_el_job_no_termina_el_script_no_se_cuelga_y_lo_informa() -> None:
    cliente = _api([_job("RUNNING")] * 50, [])
    async with cliente:
        c = await correr_caso(cliente, Caso("t", "Una tarea que no termina nunca"), timeout_s=0.01)
    assert c.veredicto == "REVISAR" and "siguió en RUNNING" in c.error


def test_la_tabla_para_el_readme_tiene_una_fila_por_corrida() -> None:
    tabla = tabla_markdown([Corrida("1. Caso", "t", job_id="b" * 32, estado="DONE", estado_final="completado", veredicto="OK", fuentes_citadas=["GD-POL-001 (a.md, pág. 1)"])])
    assert tabla.count("\n") == 3 and "GD-POL-001" in tabla and "`bbbbbbbb…`" in tabla
