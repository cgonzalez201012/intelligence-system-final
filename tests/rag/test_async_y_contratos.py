"""Lo propio de esta versión del RAG: es asíncrono, valida con Pydantic y reintenta la cuota sin bloquear."""
import asyncio
import dataclasses
import time

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ValidationError

from rag import rag_system
from rag.errores import EmbeddingsError
from rag.rag_system import RAGSystem
from rag.schemas import ConsultaRAG, FragmentoRecuperado, ResultadoRAG
from tests.rag.fakes import VectorStoreFijo, documento

CORPUS = [
    documento("El procedimiento GD-PRO-003 regula el alta de articulos nuevos", doc="proc", categoria="procedimiento", n=0),
    documento("Cuando una caja no puede cobrar se escala al responsable", doc="pol", categoria="politica", n=1),
    *[documento(f"texto de relleno numero {i}", doc=f"otro{i}", categoria="glosario", n=10 + i) for i in range(6)],
]


class RecuperadorInestable(BaseRetriever):
    """Falla con el error que da Gemini al agotarse la cuota las primeras `fallos` veces y después responde."""

    fallos: int
    mensaje: str = "429 RESOURCE_EXHAUSTED: quota exceeded"
    llamadas: int = 0

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        self.llamadas += 1
        if self.llamadas <= self.fallos:
            raise RuntimeError(self.mensaje)
        return [CORPUS[1]]


class RecuperadorLento(BaseRetriever):
    """Su implementación es SÍNCRONA y tarda: sirve para comprobar que no congela el event loop."""

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        time.sleep(0.3)
        return [CORPUS[1]]


def _con_ensemble(rag: RAGSystem, recuperador: BaseRetriever) -> RAGSystem:
    rag._cache[None] = dataclasses.replace(rag._cache[None], ensemble=recuperador)
    return rag


# --------------------------------------------------------------------------- contratos Pydantic
async def test_consultar_valida_la_entrada_y_devuelve_un_resultado_tipado_con_su_latencia():
    rag = RAGSystem(VectorStoreFijo([CORPUS[1]]), CORPUS)
    resultado = await rag.consultar(ConsultaRAG(consulta="GD-PRO-003", k=2))
    assert isinstance(resultado, ResultadoRAG)
    assert resultado.consulta == "GD-PRO-003" and resultado.modo == "hibrido" and resultado.categoria is None
    assert 0 < len(resultado.fragmentos) <= 2 and resultado.latencia_ms >= 0
    assert all(isinstance(f, FragmentoRecuperado) for f in resultado.fragmentos)


@pytest.mark.parametrize(
    "datos",
    [
        {"consulta": ""},
        {"consulta": "ab"},
        {"consulta": "x" * 501},
        {"consulta": "valida", "k": 0},
        {"consulta": "valida", "k": 11},
        {"consulta": "valida", "categoria": "inventada"},
        {"consulta": "valida", "modo": "magico"},
        {"consulta": "valida", "campo_extra": 1},
    ],
)
def test_una_consulta_invalida_se_rechaza_antes_de_llegar_a_pinecone(datos):
    with pytest.raises(ValidationError):
        ConsultaRAG(**datos)


def test_la_consulta_se_limpia_de_espacios():
    assert ConsultaRAG(consulta="   plazo de incidentes   ").consulta == "plazo de incidentes"


def test_el_fragmento_arma_una_cita_corta_y_estable():
    fragmento = FragmentoRecuperado(
        posicion=1, documento_id="01-politica", fuente="01-politica.md", pagina=1, codigo="GD-POL-001", chunk_id="a#1", texto="t"
    )
    assert fragmento.cita == "GD-POL-001 (01-politica.md, pág. 1)"
    with pytest.raises(ValidationError):
        fragmento.texto = "otro"          # inmutable: nadie le cambia el texto citado


# --------------------------------------------------------------------------- reintentos sin bloquear
async def test_ante_cuota_agotada_reintenta_con_espera_creciente_y_termina_respondiendo(monkeypatch):
    esperas: list[float] = []

    async def dormir_falso(segundos: float) -> None:
        esperas.append(segundos)

    monkeypatch.setattr(rag_system.asyncio, "sleep", dormir_falso)
    inestable = RecuperadorInestable(fallos=2)
    rag = _con_ensemble(RAGSystem(VectorStoreFijo([]), CORPUS), inestable)
    fragmentos = await rag.buscar("caja que no cobra")
    assert [f.chunk_id for f in fragmentos] == ["pol#1"]
    assert inestable.llamadas == 3 and esperas == [2.0, 4.0]


async def test_si_la_cuota_no_se_recupera_falla_con_un_error_que_explica_que_hacer(monkeypatch):
    async def dormir_falso(segundos: float) -> None:
        return None

    monkeypatch.setattr(rag_system.asyncio, "sleep", dormir_falso)
    inestable = RecuperadorInestable(fallos=99)
    rag = _con_ensemble(RAGSystem(VectorStoreFijo([]), CORPUS), inestable)
    with pytest.raises(EmbeddingsError, match="cuota"):
        await rag.buscar("caja que no cobra")
    assert inestable.llamadas == rag_system.BUSQUEDA_MAX_INTENTOS


async def test_un_error_que_no_es_de_cuota_no_se_reintenta():
    inestable = RecuperadorInestable(fallos=99, mensaje="connection reset by peer")
    rag = _con_ensemble(RAGSystem(VectorStoreFijo([]), CORPUS), inestable)
    with pytest.raises(RuntimeError, match="connection reset"):
        await rag.buscar("caja que no cobra")
    assert inestable.llamadas == 1


async def test_una_busqueda_lenta_y_sincrona_no_congela_el_event_loop():
    rag = _con_ensemble(RAGSystem(VectorStoreFijo([]), CORPUS), RecuperadorLento())
    latidos = 0

    async def latido() -> None:
        nonlocal latidos
        while True:
            await asyncio.sleep(0.01)
            latidos += 1

    tarea = asyncio.create_task(latido())
    await rag.buscar("caja que no cobra")
    tarea.cancel()
    # Si buscar bloqueara el loop, el latido no habría avanzado durante los 0,3 s de la búsqueda.
    assert latidos >= 10


async def test_varias_busquedas_concurrentes_no_se_pisan():
    rag = RAGSystem(VectorStoreFijo([CORPUS[1]]), CORPUS)
    resultados = await asyncio.gather(*(rag.buscar("GD-PRO-003", modo="bm25") for _ in range(5)))
    assert all([f.chunk_id for f in r] == ["proc#0"] for r in resultados)

