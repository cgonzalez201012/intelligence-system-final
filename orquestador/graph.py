"""Grafo principal: el Supervisor y sus especialistas.

                         +--------------+
        START ---------> |  supervisor  | <----------------------+
                         +------+-------+                        |
        (arista condicional,    |                                 |
         `decidir_ruta` con     +--> investigador --+             |
         retorno `Literal`)     +--> analista -------+--> validacion
                                +--> redactor ------+
                                |
                                +--> FINISH --> END

- `supervisor` decide quien interviene. Es el unico nodo con aristas condicionales.
- Cada especialista, al terminar, pasa SIEMPRE por `validacion`, que aplica la rubrica de
  controles y deja su veredicto en el estado; despues el control vuelve al supervisor.
- El supervisor lee el veredicto y decide: pedir un refinamiento, seguir con el proximo
  especialista o cerrar.
"""

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from orquestador.agents.analyst_agent import crear_nodo_analista
from orquestador.agents.research_agent import crear_nodo_investigador
from orquestador.agents.writer_agent import crear_nodo_redactor
from orquestador.config import Configuracion
from orquestador.state import EstadoOrquestador
from orquestador.supervisor import crear_nodo_supervisor
from orquestador.validacion import nodo_validacion

NODO_SUPERVISOR = "supervisor"
NODO_INVESTIGADOR = "investigador"
NODO_ANALISTA = "analista"
NODO_REDACTOR = "redactor"
NODO_VALIDACION = "validacion"

ESPECIALISTAS = (NODO_INVESTIGADOR, NODO_ANALISTA, NODO_REDACTOR)


def decidir_ruta(estado: EstadoOrquestador) -> Literal["investigador", "analista", "redactor", "FINISH"]:
    """Funcion de las aristas condicionales: lee el destino que dejo el supervisor en el estado.

    El tipo de retorno `Literal` enumera los unicos destinos posibles; el grafo no tiene
    ninguna ruta fija entre los especialistas.
    """
    return estado["siguiente"]


def armar_grafo(
    llm_supervisor: BaseChatModel,
    llm_especialistas: BaseChatModel,
    *,
    max_pasos: int,
    max_intentos: int,
    recursion_especialista: int,
    destino_cierre: str = END,
) -> StateGraph:
    """Arma el grafo (sin compilar) con los modelos que se le pasen (reales o falsos en los tests).

    `destino_cierre` es a donde va el flujo cuando el supervisor decide FINISH. Por defecto es
    END; la API de produccion lo reemplaza por la etapa de aprobacion humana, que se cuelga
    del cierre sin tocar la logica del supervisor.
    """
    grafo = StateGraph(EstadoOrquestador)

    nodos: dict[str, Any] = {
        NODO_SUPERVISOR: crear_nodo_supervisor(llm_supervisor, max_pasos, max_intentos),
        NODO_INVESTIGADOR: crear_nodo_investigador(llm_especialistas, recursion_especialista),
        NODO_ANALISTA: crear_nodo_analista(llm_especialistas, recursion_especialista),
        NODO_REDACTOR: crear_nodo_redactor(llm_especialistas),
        NODO_VALIDACION: nodo_validacion,
    }
    for nombre, nodo in nodos.items():
        grafo.add_node(nombre, nodo)

    grafo.add_edge(START, NODO_SUPERVISOR)
    grafo.add_conditional_edges(
        NODO_SUPERVISOR,
        decidir_ruta,
        {
            "investigador": NODO_INVESTIGADOR,
            "analista": NODO_ANALISTA,
            "redactor": NODO_REDACTOR,
            "FINISH": destino_cierre,
        },
    )
    for especialista in ESPECIALISTAS:
        grafo.add_edge(especialista, NODO_VALIDACION)
    grafo.add_edge(NODO_VALIDACION, NODO_SUPERVISOR)
    return grafo


def construir_grafo(
    llm_supervisor: BaseChatModel,
    llm_especialistas: BaseChatModel,
    *,
    max_pasos: int,
    max_intentos: int,
    recursion_especialista: int,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Arma y compila el grafo del modulo 6 (termina en END), con checkpointer opcional."""
    grafo = armar_grafo(
        llm_supervisor,
        llm_especialistas,
        max_pasos=max_pasos,
        max_intentos=max_intentos,
        recursion_especialista=recursion_especialista,
    )
    return grafo.compile(name="orquestador", checkpointer=checkpointer)
