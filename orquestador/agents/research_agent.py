"""Agente Investigador: recolecta incidentes y reglas de calidad con herramientas de solo lectura.

Es el especialista de "busqueda/investigacion". Su fuente externa es la base simulada de
`repositorio.py` (datos ficticios de Mercado Austral S.A.). Sus herramientas son de solo
lectura y acotadas: no puede calcular ni escribir nada.

El contrato de salida (`Hallazgos`) se arma a partir de lo que devolvieron las
HERRAMIENTAS, no de lo que el modelo escribio. Lo que el modelo escribe solo viaja como
`resumen`. Asi, el dato que usan los demas agentes nunca depende de que el modelo lo
transcriba bien.
"""

import json
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag.errores import RagError
from rag.schemas import CategoriaDocumento, ConsultaRAG

from orquestador.agents.base import (
    construir_agente_react,
    ejecutar_agente_react,
    entrada_con_correcciones,
    es_error,
    nombres_de_herramientas,
    resultados_de,
    texto_final,
)
from orquestador.errores import ErrorConsulta
from orquestador.rag_provider import obtener_rag
from orquestador.repositorio import obtener_repositorio
from orquestador.state import (
    FUERA_DE_ALCANCE,
    Contribucion,
    EstadoOrquestador,
    FragmentoCitado,
    Hallazgos,
    IncidenteHallazgo,
    NombreAgente,
    ReglaHallazgo,
)

NOMBRE: NombreAgente = "investigador"

PROMPT_INVESTIGADOR = f"""Sos el Agente Investigador de un equipo de Gobierno de Datos de Mercado Austral S.A.,
una empresa ficticia de retail. Tu único trabajo es RECOLECTAR datos y textos de políticas con tus herramientas.
No analices, no calcules, no priorices ni recomiendes: de eso se ocupan otros agentes.

Reglas de trabajo:
1. Todo dato concreto (IDs, fechas, severidades, umbrales, mediciones) tiene que salir de las
   herramientas. Nunca lo inventes ni lo completes de memoria.
2. Hacé lo que pide la instrucción y nada más. Para cada incidente que haya que analizar o
   priorizar necesitás su ficha completa (detalle_incidente) y la regla que incumple
   (consultar_regla_calidad); el listado solo no alcanza. Tu trabajo no está terminado hasta
   tener AMBAS cosas de cada incidente: pedí las fichas y, apenas sepas qué reglas incumplen,
   consultá cada una.
   Lo que consultaste en vueltas anteriores de esta misma tarea se conserva: si la instrucción
   pide completar algo, consultá solo lo que falta.
3. Si una herramienta devuelve un error, leé el campo 'ayuda' y corregí la llamada.
4. Si la instrucción pide saber qué dice una política, un rol, un procedimiento o un término del
   glosario de Gobierno de Datos (plazos, responsables, qué hacer ante un caso), usá
   buscar_en_politicas. Lo que sepas de esos documentos tiene que salir de los fragmentos que
   devuelva, y al resumir citá el código y la fuente (por ejemplo GD-POL-001 (01-politica.md, pág. 1)).
   Si la búsqueda no trae nada relevante, decilo: no completes con lo que "debería" decir.
5. Si la instrucción no se puede responder con tus herramientas (por ejemplo, no trata de
   incidentes, reglas ni políticas de Gobierno de Datos), no llames a ninguna herramienta y respondé
   exactamente "{FUERA_DE_ALCANCE}" seguido de una frase breve que explique por qué.
6. Al terminar, respondé con un resumen breve de qué consultaste y qué encontraste (con los
   IDs). Sin opiniones ni juicios que no estén en los datos."""


# ---------- Herramientas ----------


def _a_texto(resultado: dict[str, Any]) -> str:
    """Las herramientas devuelven JSON en texto: es lo que el modelo lee."""
    return json.dumps(resultado, ensure_ascii=False)


def _error_a_texto(error: ErrorConsulta) -> str:
    return _a_texto({"error": error.mensaje, "ayuda": error.ayuda})


class BuscarIncidentesArgs(BaseModel):
    """Argumentos de buscar_incidentes_calidad."""

    model_config = ConfigDict(extra="forbid")

    maestro: str = Field(
        description="Maestro de datos a consultar. Valores validos: 'articulos', 'proveedores' o 'sucursales'."
    )
    estado: str = Field(
        default="abierto",
        description="Estado de los incidentes: 'abierto', 'en_curso', 'cerrado' o 'todos'. Por defecto 'abierto'.",
    )
    severidad: str | None = Field(
        default=None,
        description="Filtro opcional por severidad: 'critica', 'alta', 'media' o 'baja'. Si se omite trae todas.",
    )


class DetalleIncidenteArgs(BaseModel):
    """Argumentos de detalle_incidente."""

    model_config = ConfigDict(extra="forbid")

    incidente_id: str = Field(description="ID del incidente, con formato INC-NNNN (por ejemplo 'INC-1001').")


class ReglaCalidadArgs(BaseModel):
    """Argumentos de consultar_regla_calidad."""

    model_config = ConfigDict(extra="forbid")

    regla_id: str = Field(description="ID de la regla de calidad, con formato R-XXX-NN (por ejemplo 'R-ART-01').")


@tool("buscar_incidentes_calidad", args_schema=BuscarIncidentesArgs)
async def buscar_incidentes_calidad(maestro: str, estado: str = "abierto", severidad: str | None = None) -> str:
    """Lista los incidentes de calidad de datos de UN maestro de Mercado Austral S.A.

    Usala para saber CUÁNTOS incidentes hay y cuáles son (ID, titulo, severidad, estado,
    fecha de alta y dias transcurridos desde el alta). Se consulta un solo maestro por
    llamada: 'articulos', 'proveedores' o 'sucursales'. Se puede filtrar por estado
    (abierto, en_curso, cerrado o todos) y por severidad (critica, alta, media o baja).

    NO devuelve la regla incumplida, los registros afectados ni el plazo (SLA): para eso hay
    que llamar a detalle_incidente con el ID de cada incidente que interese.

    Devuelve un JSON con 'cantidad' y 'incidentes'. Si el maestro, el estado o la severidad
    no existen, devuelve un JSON con 'error' y 'ayuda' (valores validos).
    """
    try:
        resultado = await obtener_repositorio().buscar_incidentes(maestro, estado, severidad)
    except ErrorConsulta as error:
        return _error_a_texto(error)
    return _a_texto(resultado)


@tool("detalle_incidente", args_schema=DetalleIncidenteArgs)
async def detalle_incidente(incidente_id: str) -> str:
    """Devuelve la ficha completa de UN incidente de calidad de datos a partir de su ID.

    Usala cuando ya tenés el ID (formato INC-NNNN, por ejemplo INC-1001) y necesitás datos
    que el listado no trae: responsable (Data Owner), Data Steward, regla de calidad
    incumplida (regla_id), cantidad de registros afectados, plazo de resolución (SLA en
    horas hábiles segun la severidad) y fecha de corte de los datos.

    Si no conocés el ID, primero obtenelo con buscar_incidentes_calidad: no lo inventes.
    Si el ID no existe, devuelve un JSON con 'error' y 'ayuda' que lista los IDs existentes.
    """
    try:
        resultado = await obtener_repositorio().detalle_incidente(incidente_id)
    except ErrorConsulta as error:
        return _error_a_texto(error)
    return _a_texto(resultado)


@tool("consultar_regla_calidad", args_schema=ReglaCalidadArgs)
async def consultar_regla_calidad(regla_id: str) -> str:
    """Devuelve una regla de calidad de datos: qué exige, su umbral y su medicion actual.

    Usala para saber QUÉ regla incumple un incidente o si un maestro cumple el umbral de
    aceptacion. Necesita el ID de la regla (formato R-XXX-NN, por ejemplo R-ART-01); ese ID
    figura en el campo regla_id del detalle de un incidente (detalle_incidente), no en el
    listado de buscar_incidentes_calidad.

    Devuelve un JSON con nombre, descripcion, umbral_aceptacion_pct, medicion_actual_pct y
    cumple_umbral. Si el ID no existe, devuelve un JSON con 'error' y 'ayuda'.
    """
    try:
        resultado = await obtener_repositorio().regla(regla_id)
    except ErrorConsulta as error:
        return _error_a_texto(error)
    return _a_texto(resultado)


class BuscarPoliticasArgs(BaseModel):
    """Argumentos de buscar_en_politicas (los valida Pydantic antes de tocar Pinecone)."""

    consulta: str = Field(
        min_length=3,
        max_length=500,
        description="Pregunta o frase a buscar, en lenguaje natural (por ejemplo 'plazo para resolver un incidente crítico') "
        "o un código de documento ('GD-PRO-003').",
    )
    categoria: CategoriaDocumento | None = Field(
        default=None,
        description="Filtro opcional por tipo de documento: 'politica', 'roles', 'procedimiento', 'glosario' o 'general'. "
        "Si se omite busca en todos.",
    )
    k: int = Field(default=3, ge=1, le=10, description="Cuántos fragmentos traer (1 a 10). Por defecto 3.")


@tool("buscar_en_politicas", args_schema=BuscarPoliticasArgs)
async def buscar_en_politicas(consulta: str, categoria: CategoriaDocumento | None = None, k: int = 3) -> str:
    """Busca en las políticas, roles, procedimientos y glosario de Gobierno de Datos (RAG híbrido).

    Usala cuando haga falta saber QUÉ dice la documentación interna: plazos de resolución, quién es
    responsable de qué, cómo se escala un caso, qué significa un término. Combina búsqueda semántica
    y por palabras clave sobre los documentos indexados de Mercado Austral.

    Devuelve un JSON con 'fragmentos' (texto original, código del documento, fuente, página y la
    'cita' lista para usar). Cada dato que uses de esas políticas tiene que salir de esos
    fragmentos. Si no hay coincidencias devuelve 'fragmentos' vacío. Si falla, devuelve 'error' y 'ayuda'.
    """
    try:
        rag = await obtener_rag()
        resultado = await rag.consultar(ConsultaRAG(consulta=consulta, categoria=categoria, k=k))
    except RagError as error:
        return _a_texto({"error": str(error), "ayuda": "Revisá las variables de entorno del RAG y que el índice esté cargado."})
    return _a_texto(
        {
            "consulta": resultado.consulta,
            "categoria": resultado.categoria,
            "cantidad": len(resultado.fragmentos),
            "fragmentos": [{**f.model_dump(mode="json"), "cita": f.cita} for f in resultado.fragmentos],
        }
    )


HERRAMIENTAS_INVESTIGADOR: list[BaseTool] = [
    buscar_incidentes_calidad,
    detalle_incidente,
    consultar_regla_calidad,
    buscar_en_politicas,
]


# ---------- Del resultado de las herramientas al contrato `Hallazgos` ----------

_CAMPOS_LISTADO = ("id", "titulo", "severidad", "estado", "fecha_alta")
_CAMPOS_DETALLE = (
    "id",
    "maestro",
    "titulo",
    "severidad",
    "estado",
    "fecha_alta",
    "regla_id",
    "registros_afectados",
    "responsable",
    "data_steward",
    "sla_horas_habiles",
)
_CAMPOS_FRAGMENTO = ("chunk_id", "documento_id", "codigo", "fuente", "pagina", "cita", "texto")
_CAMPOS_REGLA = ("id", "maestro", "nombre", "descripcion", "umbral_aceptacion_pct", "medicion_actual_pct", "cumple_umbral")


def construir_hallazgos(mensajes: list[BaseMessage], resumen: str) -> Hallazgos:
    """Arma `Hallazgos` leyendo los resultados de las herramientas (no el texto del modelo)."""
    errores: list[str] = []
    fecha_corte: date | None = None
    incidentes: dict[str, dict[str, Any]] = {}
    reglas: dict[str, ReglaHallazgo] = {}
    politicas: dict[str, FragmentoCitado] = {}

    for resultado in resultados_de(mensajes, "buscar_incidentes_calidad"):
        if es_error(resultado):
            errores.append(f"buscar_incidentes_calidad: {resultado['error']}")
            continue
        try:
            fecha_corte = fecha_corte or date.fromisoformat(resultado["fecha_corte"])
            for item in resultado["incidentes"]:
                datos = {campo: item[campo] for campo in _CAMPOS_LISTADO}
                datos["maestro"] = resultado["maestro"]
                incidentes[item["id"]] = {**incidentes.get(item["id"], {}), **datos}
        except (KeyError, TypeError, ValueError):
            errores.append("buscar_incidentes_calidad: respuesta con formato inesperado")

    for resultado in resultados_de(mensajes, "detalle_incidente"):
        if es_error(resultado):
            errores.append(f"detalle_incidente: {resultado['error']}")
            continue
        try:
            fecha_corte = fecha_corte or date.fromisoformat(resultado["fecha_corte"])
            datos = {campo: resultado[campo] for campo in _CAMPOS_DETALLE}
            incidentes[datos["id"]] = {**incidentes.get(datos["id"], {}), **datos}
        except (KeyError, TypeError, ValueError):
            errores.append("detalle_incidente: respuesta con formato inesperado")

    for resultado in resultados_de(mensajes, "consultar_regla_calidad"):
        if es_error(resultado):
            errores.append(f"consultar_regla_calidad: {resultado['error']}")
            continue
        try:
            regla = ReglaHallazgo(**{campo: resultado[campo] for campo in _CAMPOS_REGLA})
        except (KeyError, TypeError, ValidationError):
            errores.append("consultar_regla_calidad: respuesta con formato inesperado")
            continue
        reglas[regla.id] = regla

    for resultado in resultados_de(mensajes, "buscar_en_politicas"):
        if es_error(resultado):
            errores.append(f"buscar_en_politicas: {resultado['error']}")
            continue
        try:
            for item in resultado["fragmentos"]:
                fragmento = FragmentoCitado(**{campo: item[campo] for campo in _CAMPOS_FRAGMENTO})
                politicas[fragmento.chunk_id] = fragmento
        except (KeyError, TypeError, ValidationError):
            errores.append("buscar_en_politicas: respuesta con formato inesperado")

    validos: list[IncidenteHallazgo] = []
    for datos in incidentes.values():
        try:
            validos.append(IncidenteHallazgo(**datos))
        except ValidationError:
            errores.append(f"incidente {datos.get('id', '?')}: datos con formato inesperado")

    return Hallazgos(
        fecha_corte=fecha_corte,
        incidentes=validos,
        reglas=list(reglas.values()),
        politicas=list(politicas.values()),
        consultas_con_error=errores,
        herramientas_usadas=nombres_de_herramientas(mensajes, {h.name for h in HERRAMIENTAS_INVESTIGADOR}),
        fuera_de_alcance=resumen.strip().startswith(FUERA_DE_ALCANCE),
        resumen=resumen,
    )


# ---------- Nodo del grafo ----------

NodoAsincrono = Callable[[EstadoOrquestador, RunnableConfig], Awaitable[dict[str, Any]]]


def crear_nodo_investigador(llm: BaseChatModel, recursion_limit: int) -> NodoAsincrono:
    """Devuelve el nodo del grafo principal que corre al investigador."""
    subgrafo = construir_agente_react(llm, HERRAMIENTAS_INVESTIGADOR, PROMPT_INVESTIGADOR, NOMBRE)

    async def nodo_investigador(estado: EstadoOrquestador, config: RunnableConfig) -> dict[str, Any]:
        # Contexto aislado: solo la instruccion (y las correcciones, si es un refinamiento).
        entrada = entrada_con_correcciones(estado["instruccion"], estado, NOMBRE)
        corrida = await ejecutar_agente_react(subgrafo, entrada, recursion_limit, config)
        resumen = texto_final(corrida.mensajes)
        de_esta_corrida = construir_hallazgos(corrida.mensajes, resumen)
        contribucion = Contribucion(
            agente=NOMBRE,
            paso=estado["pasos"],
            resumen=resumen or (corrida.error or "Sin respuesta."),
            herramientas_usadas=de_esta_corrida.herramientas_usadas,
            error=corrida.error,
        )
        # Un refinamiento completa lo recolectado antes; no lo reemplaza.
        anteriores = estado["hallazgos"]
        hallazgos = anteriores.fusionar(de_esta_corrida) if anteriores is not None else de_esta_corrida
        return {
            "hallazgos": hallazgos,
            "ultimo_agente": NOMBRE,
            "contribuciones": [contribucion],
            "messages": [AIMessage(content=contribucion.resumen, name=NOMBRE)],
        }

    return nodo_investigador
