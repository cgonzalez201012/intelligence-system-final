"""De punta a punta con el RAG: POST /tasks -> cola -> worker -> grafo (investigador con RAG) -> GET con las fuentes."""

import httpx
import pytest

from app.worker import procesar_mensaje
from orquestador import rag_provider
from rag.rag_system import RAGSystem
from tests.fakes import SupervisorFalso, decidir, especialistas_falsos, pedir, responder
from tests.rag.fakes import VectorStoreFijo, documento

CORPUS = [
    documento("El plazo para resolver un incidente critico es de 4 horas habiles", doc="01-politica", categoria="politica", n=0),
    *[documento(f"relleno {i}", doc=f"otro{i}", categoria="glosario", n=10 + i) for i in range(6)],
]


@pytest.fixture(autouse=True)
def rag_inyectado():
    rag_provider.fijar_rag(RAGSystem(VectorStoreFijo([CORPUS[0]]), CORPUS))
    yield
    rag_provider.fijar_rag(None)


def _supervisor() -> SupervisorFalso:
    return SupervisorFalso(guion=[decidir("investigador", "Buscá la política."), decidir("redactor", "Escribí."), decidir("FINISH")])


async def _correr(cliente: httpx.AsyncClient, armar_servicios, redactor: str) -> dict:
    modelos = especialistas_falsos(
        investigador=[pedir("buscar_en_politicas", consulta="plazo para resolver un incidente critico"), responder("Encontré GD-POL-001.")],
        redactor=[responder(redactor)],
    )
    servicios = armar_servicios(_supervisor(), modelos)
    creado = (await cliente.post("/tasks", json={"tarea": "¿Cuál es el plazo para resolver un incidente crítico?"})).json()
    mensaje = await servicios.almacen.tomar_de_la_cola(0.1)
    assert mensaje is not None
    await procesar_mensaje(mensaje, servicios)
    return (await cliente.get(f"/tasks/{creado['job_id']}")).json()


async def test_la_respuesta_de_la_api_trae_las_fuentes_del_rag_que_cito(cliente: httpx.AsyncClient, armar_servicios) -> None:
    job = await _correr(cliente, armar_servicios, "Según GD-POL-001 (01-politica.md, pág. 1) el plazo es de 4 horas hábiles.")
    assert job["estado"] == "DONE" and job["resultado"]["estado_final"] == "completado"
    assert job["resultado"]["fuentes_citadas"] == ["GD-POL-001 (01-politica.md, pág. 1)"]
    assert job["resultado"]["agentes_que_intervinieron"] == ["investigador", "redactor"]
    assert job["resultado"]["acciones"]["propuestas"] == []  # una consulta de políticas no escala nada


async def test_si_el_informe_cita_una_politica_inventada_nunca_se_entrega_como_completado(cliente: httpx.AsyncClient, armar_servicios) -> None:
    """El redactor inventa GD-POL-009: la validación lo rechaza y, al no corregirse, el job no termina como completado."""
    job = await _correr(cliente, armar_servicios, "Según GD-POL-009 el plazo es de 4 horas hábiles.")
    assert job["estado"] == "DONE" and job["resultado"]["estado_final"] != "completado"
    assert job["resultado"]["fuentes_citadas"] == []
