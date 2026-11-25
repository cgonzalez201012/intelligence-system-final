"""Observabilidad: cada ejecucion deja una traza con spans por nodo, y el servicio no depende de Phoenix."""

import json
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from langgraph.graph.state import CompiledStateGraph
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from openinference.instrumentation.langchain import LangChainInstrumentor

from app import observability
from app.observability import (
    etiquetas_de_traza,
    iniciar_observabilidad,
    instrumentar_langchain,
    silenciar_trazas_propias_del_sdk,
)
from app.settings import Ajustes
from orquestador.state import estado_inicial
from tests.fakes import SupervisorFalso, decidir, especialistas_falsos, guion_analista, guion_investigador, informe


@pytest.fixture
def exportador() -> Iterator[InMemorySpanExporter]:
    """Captura en memoria los spans que genera el instrumentador (en lugar de mandarlos a Phoenix)."""
    exportador = InMemorySpanExporter()
    proveedor = TracerProvider()
    proveedor.add_span_processor(SimpleSpanProcessor(exportador))
    instrumentar_langchain(proveedor)
    yield exportador
    LangChainInstrumentor().uninstrument()


def _atributos(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


async def test_cada_nodo_del_grafo_y_cada_llamada_al_modelo_generan_spans_de_la_misma_traza(
    exportador: InMemorySpanExporter, armar_grafo_produccion: Callable[..., CompiledStateGraph]
) -> None:
    supervisor = SupervisorFalso(guion=[decidir("investigador", "r"), decidir("analista", "c"), decidir("redactor", "e"), decidir("FINISH")])
    grafo = armar_grafo_produccion(supervisor, especialistas_falsos(guion_investigador(), guion_analista(), [informe("INC-1001", "INC-1002")]))
    config = {"configurable": {"thread_id": "job-traza"}, "recursion_limit": 60, "run_name": "tarea"}
    with etiquetas_de_traza("job-traza", "m7-test"):
        await grafo.ainvoke(estado_inicial("tarea"), config)  # type: ignore[arg-type]

    spans = exportador.get_finished_spans()
    nombres = {span.name for span in spans}
    for nodo in ("supervisor", "investigador", "analista", "redactor", "validacion", "proponer_acciones", "aprobacion_humana"):
        assert nodo in nombres, f"falta el span del nodo {nodo}"
    assert any(_atributos(s).get("openinference.span.kind") == "LLM" for s in spans)  # llamadas al modelo
    assert any(_atributos(s).get("openinference.span.kind") == "TOOL" for s in spans)  # herramientas
    assert len({s.context.trace_id for s in spans}) == 1  # todo es UNA traza: la de la tarea


async def test_las_trazas_llevan_el_job_id_como_sesion_y_la_version_de_los_prompts(
    exportador: InMemorySpanExporter, armar_grafo_produccion: Callable[..., CompiledStateGraph]
) -> None:
    supervisor = SupervisorFalso(guion=[decidir("investigador", "r"), decidir("analista", "c"), decidir("redactor", "e"), decidir("FINISH")])
    grafo = armar_grafo_produccion(supervisor, especialistas_falsos(guion_investigador(), guion_analista(), [informe("INC-1001", "INC-1002")]))
    with etiquetas_de_traza("job-123", "m7-v9"):
        await grafo.ainvoke(estado_inicial("tarea"), {"configurable": {"thread_id": "job-123"}, "recursion_limit": 60})
    spans = exportador.get_finished_spans()
    assert spans
    for span in spans:
        atributos = _atributos(span)
        assert atributos["session.id"] == "job-123"
        assert atributos["llm.prompt_template.version"] == "m7-v9"
        assert json.loads(atributos["metadata"])["job_id"] == "job-123"


async def test_dos_jobs_concurrentes_no_mezclan_sus_sesiones(
    exportador: InMemorySpanExporter, armar_grafo_produccion: Callable[..., CompiledStateGraph]
) -> None:
    """Cada job corre en su tarea asincrona: el contexto de la traza no se filtra de uno al otro."""
    import asyncio

    async def correr(job_id: str) -> None:
        supervisor = SupervisorFalso(guion=[decidir("investigador", "r"), decidir("analista", "c"), decidir("redactor", "e"), decidir("FINISH")])
        grafo = armar_grafo_produccion(supervisor, especialistas_falsos(guion_investigador(), guion_analista(), [informe("INC-1001", "INC-1002")]))
        with etiquetas_de_traza(job_id, "m7-test"):
            await grafo.ainvoke(estado_inicial("tarea"), {"configurable": {"thread_id": job_id}, "recursion_limit": 60})

    await asyncio.gather(*[correr(f"job-{i}") for i in range(3)])
    por_traza: dict[int, set[str]] = {}
    for span in exportador.get_finished_spans():
        por_traza.setdefault(span.context.trace_id, set()).add(str(_atributos(span)["session.id"]))
    assert len(por_traza) == 3
    assert all(len(sesiones) == 1 for sesiones in por_traza.values())  # una sesion por traza, sin mezclas


def _ajustes(observabilidad: bool) -> Ajustes:
    return Ajustes(
        redis_url="redis://x", observabilidad=observabilidad, phoenix_endpoint="http://localhost:6006/v1/traces",
        phoenix_proyecto="p", version_prompts="v", worker_concurrencia=1, ttl_jobs_segundos=60, ttl_checkpoints_minutos=1,
    )


def test_con_la_observabilidad_desactivada_no_se_conecta_a_nada() -> None:
    assert iniciar_observabilidad(_ajustes(False)) is False


def test_si_phoenix_no_se_puede_iniciar_el_servicio_sigue_sin_trazas(monkeypatch: pytest.MonkeyPatch) -> None:
    def explota(**_: Any) -> None:
        raise RuntimeError("phoenix no disponible")

    monkeypatch.setattr("phoenix.otel.register", explota)
    assert iniciar_observabilidad(_ajustes(True)) is False  # no tira abajo al worker


def test_el_modulo_expone_las_funciones_que_usa_el_worker() -> None:
    assert callable(observability.iniciar_observabilidad) and callable(observability.etiquetas_de_traza)


def test_se_apagan_los_spans_propios_del_sdk_de_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Esos spans salen como trazas sueltas (sin padre) y falsean el p95 del dashboard."""
    monkeypatch.delenv("ANTHROPIC_OPEN_TELEMETRY", raising=False)
    silenciar_trazas_propias_del_sdk()
    import os

    assert os.environ["ANTHROPIC_OPEN_TELEMETRY"] == "false"


def test_al_iniciar_la_observabilidad_el_sdk_queda_sin_spans_propios(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import sys
    import types

    monkeypatch.delenv("ANTHROPIC_OPEN_TELEMETRY", raising=False)
    falso = types.ModuleType("phoenix.otel")
    falso.register = lambda **_: TracerProvider()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "phoenix.otel", falso)
    monkeypatch.setattr(observability, "instrumentar_langchain", lambda proveedor: None)
    assert iniciar_observabilidad(Ajustes.desde_entorno()) is True
    assert os.environ["ANTHROPIC_OPEN_TELEMETRY"] == "false"
