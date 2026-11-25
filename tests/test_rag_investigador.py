"""El investigador con RAG: la herramienta, el contrato `Hallazgos.politicas` y la validación de citas."""

import json

import pytest
from langchain_core.messages import BaseMessage, ToolMessage

from orquestador import rag_provider
from orquestador.agents.research_agent import HERRAMIENTAS_INVESTIGADOR, buscar_en_politicas, construir_hallazgos
from orquestador.state import FragmentoCitado, Hallazgos
from orquestador.validacion import validar_investigador, validar_redactor
from rag.errores import ConfiguracionError
from rag.rag_system import RAGSystem
from tests.fakes import analisis_de_datos, hallazgos_de_datos
from tests.rag.fakes import VectorStoreFijo, documento

CORPUS = [
    documento("El plazo para resolver un incidente critico es de 4 horas habiles", doc="01-politica", categoria="politica", n=0),
    documento("El Data Steward escala al Data Owner cuando se vence el plazo", doc="02-roles", categoria="roles", n=1),
    *[documento(f"relleno {i}", doc=f"otro{i}", categoria="glosario", n=10 + i) for i in range(6)],
]


@pytest.fixture(autouse=True)
def rag_inyectado():
    rag_provider.fijar_rag(RAGSystem(VectorStoreFijo([CORPUS[0]]), CORPUS))
    yield
    rag_provider.fijar_rag(None)


def _tool(nombre: str, contenido: object) -> ToolMessage:
    return ToolMessage(content=json.dumps(contenido), name=nombre, tool_call_id="x")


async def _buscar(**args: object) -> dict:
    return json.loads(await buscar_en_politicas.ainvoke(args))


# --------------------------------------------------------------------------- la herramienta
async def test_la_herramienta_devuelve_fragmentos_con_cita_lista_para_usar():
    resultado = await _buscar(consulta="plazo para resolver un incidente critico", k=2)
    assert resultado["cantidad"] == len(resultado["fragmentos"]) >= 1
    primero = resultado["fragmentos"][0]
    assert {"chunk_id", "documento_id", "codigo", "fuente", "texto", "cita"} <= primero.keys()
    assert primero["fuente"] in primero["cita"]


async def test_la_herramienta_acepta_filtro_por_categoria():
    resultado = await _buscar(consulta="Data Steward escala al Data Owner", categoria="roles")
    assert resultado["categoria"] == "roles"      # el filtro viaja hasta el RAG (su efecto sobre Pinecone lo prueban tests/rag)


@pytest.mark.parametrize("args", [{"consulta": "ab"}, {"consulta": "valida", "k": 11}, {"consulta": "valida", "categoria": "inventada"}])
async def test_argumentos_invalidos_se_rechazan_antes_de_buscar(args):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        await buscar_en_politicas.ainvoke(args)


async def test_si_falta_configuracion_la_herramienta_devuelve_error_con_ayuda(monkeypatch):
    rag_provider.fijar_rag(None)
    for variable in ("PINECONE_API_KEY", "INDEX_NAME", "GOOGLE_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(rag_provider, "load_dotenv", lambda *a, **k: None)
    resultado = await _buscar(consulta="plazo de incidentes")
    assert "PINECONE_API_KEY" in resultado["error"] and resultado["ayuda"]


async def test_el_rag_se_arma_una_sola_vez_aunque_lleguen_consultas_concurrentes(monkeypatch):
    import asyncio

    rag_provider.fijar_rag(None)
    armados = 0

    async def fabricar(cfg):
        nonlocal armados
        armados += 1
        await asyncio.sleep(0.05)
        return RAGSystem(VectorStoreFijo([]), CORPUS)

    monkeypatch.setattr(rag_provider, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(rag_provider.rag_config, "cargar_configuracion", lambda **k: object())
    monkeypatch.setattr(rag_provider.RAGSystem, "desde_configuracion", staticmethod(fabricar))
    sistemas = await asyncio.gather(*(rag_provider.obtener_rag() for _ in range(5)))
    assert armados == 1 and all(s is sistemas[0] for s in sistemas)


def test_la_herramienta_esta_entre_las_del_investigador():
    assert buscar_en_politicas in HERRAMIENTAS_INVESTIGADOR


# --------------------------------------------------------------------------- del resultado de la herramienta a Hallazgos
def _fragmento(codigo="GD-POL-001", n=1) -> dict:
    return {
        "chunk_id": f"01-politica#{n}", "documento_id": "01-politica", "codigo": codigo, "fuente": "01-politica.md",
        "pagina": 1, "cita": f"{codigo} (01-politica.md, pág. 1)", "texto": "El plazo es de 4 horas",
    }


def test_los_hallazgos_incluyen_los_fragmentos_que_devolvio_el_rag():
    mensajes: list[BaseMessage] = [_tool("buscar_en_politicas", {"cantidad": 1, "fragmentos": [_fragmento()]})]
    hallazgos = construir_hallazgos(mensajes, "listo")
    assert [p.codigo for p in hallazgos.politicas] == ["GD-POL-001"]
    assert hallazgos.codigos_de_politicas() == {"GD-POL-001"}


def test_un_error_del_rag_o_un_formato_raro_quedan_registrados_sin_romper():
    hallazgos = construir_hallazgos(
        [_tool("buscar_en_politicas", {"error": "Faltan variables", "ayuda": "x"}), _tool("buscar_en_politicas", {"algo": 1})], ""
    )
    assert hallazgos.politicas == []
    assert len(hallazgos.consultas_con_error) == 2


def test_fusionar_junta_fragmentos_sin_duplicar_por_chunk_id():
    a = Hallazgos(politicas=[FragmentoCitado(**_fragmento(n=1))])
    b = Hallazgos(politicas=[FragmentoCitado(**_fragmento(n=1)), FragmentoCitado(**_fragmento("GD-ROL-001", n=2))])
    assert sorted(p.chunk_id for p in a.fusionar(b).politicas) == ["01-politica#1", "01-politica#2"]


def test_una_consulta_solo_de_politicas_no_exige_incidentes_ni_reglas():
    hallazgos = Hallazgos(politicas=[FragmentoCitado(**_fragmento())], herramientas_usadas=["buscar_en_politicas"])
    assert validar_investigador(hallazgos) == []


# --------------------------------------------------------------------------- validación de citas del informe
def test_el_informe_puede_citar_una_politica_que_el_rag_recupero():
    hallazgos = Hallazgos(politicas=[FragmentoCitado(**_fragmento())])
    assert validar_redactor(hallazgos, None, "Según GD-POL-001 (01-politica.md, pág. 1) el plazo es de 4 horas.") == []


def test_el_informe_no_puede_citar_una_politica_que_no_se_recupero():
    hallazgos = Hallazgos(politicas=[FragmentoCitado(**_fragmento())])
    problemas = validar_redactor(hallazgos, None, "Según GD-POL-009 el plazo es de 4 horas.")
    assert len(problemas) == 1 and "GD-POL-009" in problemas[0]


def test_sin_hallazgos_cualquier_politica_citada_es_un_problema():
    assert validar_redactor(None, None, "Dice GD-PRO-003.")


def test_la_validacion_de_incidentes_sigue_igual_con_politicas_de_por_medio():
    hallazgos = hallazgos_de_datos()
    con_politicas = hallazgos.fusionar(Hallazgos(politicas=[FragmentoCitado(**_fragmento())]))
    texto = "INC-1001 e INC-1002 (prioridad INC-1001). Ver GD-POL-001."
    assert validar_redactor(con_politicas, analisis_de_datos(hallazgos), texto) == []


def test_si_el_supervisor_saltea_al_analista_el_informe_igual_debe_cubrir_los_incidentes_recolectados():
    hallazgos = hallazgos_de_datos()   # INC-1001 e INC-1002 detallados
    problemas = validar_redactor(hallazgos, None, "Hay incidentes críticos en artículos.")
    assert any("INC-1001" in p and "INC-1002" in p for p in problemas)
    assert validar_redactor(hallazgos, None, "Cubro INC-1001 y INC-1002.") == []


def test_una_consulta_solo_de_politicas_sin_incidentes_no_se_ve_afectada_por_esa_cobertura():
    hallazgos = Hallazgos(politicas=[FragmentoCitado(**_fragmento())])
    assert validar_redactor(hallazgos, None, "El plazo está en GD-POL-001 (01-politica.md, pág. 1).") == []
