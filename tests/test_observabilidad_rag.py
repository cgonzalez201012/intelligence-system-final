"""El RAG en Phoenix: la búsqueda queda dentro de la traza de la tarea, con los fragmentos recuperados."""

import json
from collections.abc import Iterator

import pytest
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.observability import etiquetas_de_traza, instrumentar_langchain
from orquestador import rag_provider
from orquestador.state import estado_inicial
from rag.rag_system import RAGSystem
from tests.fakes import SupervisorFalso, decidir, especialistas_falsos, pedir, responder
from tests.rag.fakes import VectorStoreFijo, documento

CORPUS = [
    documento("El plazo para resolver un incidente critico es de 4 horas habiles", doc="01-politica", categoria="politica", n=0),
    *[documento(f"relleno {i}", doc=f"otro{i}", categoria="glosario", n=10 + i) for i in range(6)],
]


@pytest.fixture
def exportador() -> Iterator[InMemorySpanExporter]:
    exportador = InMemorySpanExporter()
    proveedor = TracerProvider()
    proveedor.add_span_processor(SimpleSpanProcessor(exportador))
    instrumentar_langchain(proveedor)
    rag_provider.fijar_rag(RAGSystem(VectorStoreFijo([CORPUS[0]]), CORPUS))
    yield exportador
    rag_provider.fijar_rag(None)
    LangChainInstrumentor().uninstrument()


def _attrs(span: ReadableSpan) -> dict:
    return dict(span.attributes or {})


async def _correr_tarea(armar_grafo_produccion, exportador: InMemorySpanExporter) -> tuple[dict, list[ReadableSpan]]:
    supervisor = SupervisorFalso(guion=[decidir("investigador", "Buscá la política."), decidir("redactor", "Escribí."), decidir("FINISH")])
    modelos = especialistas_falsos(
        investigador=[pedir("buscar_en_politicas", consulta="plazo para resolver un incidente critico"), responder("Encontré la política GD-POL-001.")],
        redactor=[responder("Según GD-POL-001 (01-politica.md, pág. 1) el plazo es de 4 horas hábiles.")],
    )
    grafo = armar_grafo_produccion(supervisor, modelos)
    config = {"configurable": {"thread_id": "job-rag"}, "recursion_limit": 60, "run_name": "tarea"}
    with etiquetas_de_traza("job-rag", "m7-test"):
        final = await grafo.ainvoke(estado_inicial("¿Cuál es el plazo para resolver un incidente crítico?"), config)  # type: ignore[arg-type]
    return final, list(exportador.get_finished_spans())


async def test_la_busqueda_del_rag_queda_en_la_misma_traza_que_la_tarea_sin_trazas_sueltas(
    exportador: InMemorySpanExporter, armar_grafo_produccion
) -> None:
    final, spans = await _correr_tarea(armar_grafo_produccion, exportador)
    assert final["estado_final"] == "completado"  # y la cita quedó respaldada: el validador la aprobó
    assert len({s.context.trace_id for s in spans}) == 1  # UNA traza: una búsqueda no deja trazas huérfanas en Phoenix
    herramienta = next(s for s in spans if s.name == "buscar_en_politicas")
    assert _attrs(herramienta)["openinference.span.kind"] == "TOOL"


async def test_el_span_del_recuperador_muestra_los_fragmentos_con_su_fuente(
    exportador: InMemorySpanExporter, armar_grafo_produccion
) -> None:
    _, spans = await _correr_tarea(armar_grafo_produccion, exportador)
    recuperadores = [s for s in spans if _attrs(s).get("openinference.span.kind") == "RETRIEVER" and "retrieval.documents.0.document.content" in _attrs(s)]
    assert recuperadores, "no hay ningún span RETRIEVER con documentos"
    span = recuperadores[0]
    assert span.parent is not None  # cuelga de la traza de la tarea (no es un span suelto)
    a = _attrs(span)
    assert "4 horas" in a["retrieval.documents.0.document.content"]
    meta = json.loads(a["retrieval.documents.0.document.metadata"])
    assert meta["codigo"] == "GD-POL-001" and meta["fuente"] == "01-politica.md" and meta["chunk_id"] == "01-politica#0"


async def test_la_traza_lleva_el_job_id_como_sesion_tambien_en_la_busqueda(
    exportador: InMemorySpanExporter, armar_grafo_produccion
) -> None:
    _, spans = await _correr_tarea(armar_grafo_produccion, exportador)
    herramienta = next(s for s in spans if s.name == "buscar_en_politicas")
    assert _attrs(herramienta)["session.id"] == "job-rag"
