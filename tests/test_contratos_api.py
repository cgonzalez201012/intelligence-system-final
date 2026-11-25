"""Contratos Pydantic de la API: lo que entra, lo que sale y lo que documenta OpenAPI."""

import httpx
import pytest
from pydantic import ValidationError

from app.main import crear_app
from app.schemas import AprobacionIn, AprobacionPendiente, ResultadoTarea, TareaIn
from app.worker import resultado_desde_estado
from orquestador.state import FragmentoCitado, Hallazgos


@pytest.fixture
def openapi() -> dict:
    return crear_app().openapi()


def test_todas_las_rutas_documentan_su_respuesta_y_sus_errores(openapi):
    rutas = openapi["paths"]
    assert set(rutas) == {"/tasks", "/tasks/{job_id}", "/tasks/{job_id}/approve", "/health"}
    assert set(rutas["/tasks"]["post"]["responses"]) >= {"202", "422", "503"}
    assert set(rutas["/tasks/{job_id}"]["get"]["responses"]) >= {"200", "404", "503"}
    assert set(rutas["/tasks/{job_id}/approve"]["post"]["responses"]) >= {"202", "404", "409", "422", "503"}
    assert "503" in rutas["/health"]["get"]["responses"]


def test_ninguna_respuesta_exitosa_es_un_objeto_libre(openapi):
    """Con response_model en cada ruta, el esquema de salida está tipado (nada de `additionalProperties: true`)."""
    for ruta, metodos in openapi["paths"].items():
        for metodo, operacion in metodos.items():
            exito = next(c for c in operacion["responses"] if c.startswith("2"))
            esquema = operacion["responses"][exito]["content"]["application/json"]["schema"]
            assert "$ref" in esquema, f"{metodo.upper()} {ruta} devuelve un esquema sin tipar"


def test_los_cuerpos_de_entrada_traen_ejemplo_para_probar_en_docs(openapi):
    esquemas = openapi["components"]["schemas"]
    assert "example" in esquemas["TareaIn"] and "example" in esquemas["AprobacionIn"]


def test_el_estado_del_job_expone_la_aprobacion_pendiente_tipada(openapi):
    esquemas = openapi["components"]["schemas"]
    assert "AprobacionPendiente" in esquemas and "ErrorApi" in esquemas and "Salud" in esquemas
    assert set(esquemas["ResultadoTarea"]["properties"]) >= {"estado_final", "respuesta", "fuentes_citadas", "acciones"}


@pytest.mark.parametrize(
    "cuerpo",
    [{"tarea": "corta"}, {"tarea": "x" * 2001}, {"tarea": "Una tarea válida y larga", "extra": 1}, {}, {"tarea": 123456789012}],
)
async def test_una_tarea_invalida_se_rechaza_con_422(cliente: httpx.AsyncClient, cuerpo):
    assert (await cliente.post("/tasks", json=cuerpo)).status_code == 422


@pytest.mark.parametrize(
    "cuerpo",
    [{}, {"aprobada": "tal vez", "revisor": "Cris"}, {"aprobada": True, "revisor": ""}, {"aprobada": True, "revisor": "Cris", "x": 1}],
)
async def test_una_decision_invalida_se_rechaza_con_422(cliente: httpx.AsyncClient, cuerpo):
    assert (await cliente.post("/tasks/" + "a" * 32 + "/approve", json=cuerpo)).status_code == 422


def test_la_tarea_se_normaliza_y_la_decision_es_inmutable():
    assert TareaIn(tarea="  Priorizá   los incidentes  ").tarea == "Priorizá los incidentes"
    decision = AprobacionIn(aprobada=True, revisor="Cris")
    with pytest.raises(ValidationError):
        decision.aprobada = False


def test_la_aprobacion_pendiente_exige_acciones_completas():
    with pytest.raises(ValidationError):
        AprobacionPendiente(tipo="aprobacion_requerida", motivo="m", informe="i", acciones=[{"incidente_id": "INC-1001"}])


def test_el_estado_final_solo_admite_los_valores_del_supervisor():
    with pytest.raises(ValidationError):
        ResultadoTarea(estado_final="inventado", respuesta="x")


def _fragmento(codigo: str) -> FragmentoCitado:
    return FragmentoCitado(
        chunk_id=f"{codigo}#1", documento_id="d", codigo=codigo, fuente="d.md", pagina=1, cita=f"{codigo} (d.md, pág. 1)", texto="t"
    )


def test_el_resultado_lista_solo_las_politicas_que_la_respuesta_cita():
    hallazgos = Hallazgos(politicas=[_fragmento("GD-POL-001"), _fragmento("GD-ROL-001")])
    resultado = resultado_desde_estado(
        {"estado_final": "completado", "respuesta_final": "Según GD-POL-001 el plazo es 4 h.", "hallazgos": hallazgos}
    )
    assert resultado.fuentes_citadas == ["GD-POL-001 (d.md)"]  # nombró solo el código: no se inventa la página


def test_sin_hallazgos_no_hay_fuentes():
    assert resultado_desde_estado({"estado_final": "completado", "respuesta_final": "ok"}).fuentes_citadas == []


def test_el_openapi_publicado_en_docs_coincide_con_el_codigo():
    from scripts.exportar_openapi import RUTA, contrato

    assert RUTA.read_text(encoding="utf-8") == contrato(), "corré `python scripts/exportar_openapi.py`"


def _frag(codigo: str, fuente: str, pagina: int) -> FragmentoCitado:
    return FragmentoCitado(
        chunk_id=f"{fuente}#{pagina}", documento_id=fuente, codigo=codigo, fuente=fuente, pagina=pagina,
        cita=f"{codigo} ({fuente}, pág. {pagina})", texto="t",
    )


def test_las_fuentes_citadas_listan_la_pagina_solo_si_la_respuesta_la_cito():
    from app.worker import fuentes_citadas

    politicas = [_frag("GD-POL-001", "01.md", 1), _frag("GD-POL-001", "01.md", 3), _frag("GD-ROL-001", "02.md", 1)]
    # cita exacta de UNA pagina: no se listan las otras páginas recuperadas del mismo documento
    assert fuentes_citadas(politicas, "Según GD-POL-001 (01.md, pág. 3) el plazo es 4 h.") == ["GD-POL-001 (01.md, pág. 3)"]
    # solo el código: se lista el documento, sin inventar página
    assert fuentes_citadas(politicas, "Ver GD-ROL-001 para los responsables.") == ["GD-ROL-001 (02.md)"]
    # un código que no se nombra no aparece, y un código parecido (GD-POL-0011) no cuenta
    assert fuentes_citadas(politicas, "Ver GD-POL-0011.") == []
    assert fuentes_citadas([], "Ver GD-POL-001") == []
