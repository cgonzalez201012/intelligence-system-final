"""Los checkpoints de Redis guardan JSON: el estado tiene que volver IDENTICO, con sus modelos Pydantic.

Es el riesgo mas facil de pasar por alto: con InMemorySaver todo anda, pero con Redis, por seguridad,
LangGraph solo reconstruye las clases que se le declaran. Sin la lista, un job retomado desde Redis
(por ejemplo, tras aprobarlo) llegaria al nodo con diccionarios en lugar de `Hallazgos` o `Analisis`.
"""

import inspect
from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from langgraph.graph.state import CompiledStateGraph

import orquestador.state as estado
from app.graph import serializador_del_estado
from orquestador.state import CLASES_DEL_ESTADO, estado_inicial
from tests.fakes import SupervisorFalso, decidir, especialistas_falsos, guion_analista, guion_investigador, informe


def test_la_lista_de_clases_serializables_incluye_todos_los_modelos_del_estado() -> None:
    modelos = {
        clase
        for _, clase in inspect.getmembers(estado, inspect.isclass)
        if issubclass(clase, estado._Modelo) and clase is not estado._Modelo  # noqa: SLF001
    }
    assert modelos == set(CLASES_DEL_ESTADO)  # si se agrega un modelo nuevo al estado y se olvida declararlo, falla aca


async def _estado_pausado(armar: Callable[..., CompiledStateGraph]) -> dict[str, Any]:
    supervisor = SupervisorFalso(
        guion=[decidir("investigador", "r"), decidir("analista", "c"), decidir("redactor", "e"), decidir("FINISH")]
    )
    grafo = armar(supervisor, especialistas_falsos(guion_investigador(), guion_analista(), [informe("INC-1001", "INC-1002")]))
    config: dict[str, Any] = {"configurable": {"thread_id": "t"}, "recursion_limit": 60}
    await grafo.ainvoke(estado_inicial("tarea"), config)
    foto = await grafo.aget_state(config)  # type: ignore[arg-type]
    return dict(foto.values)


async def test_el_estado_completo_vuelve_identico_con_el_serializador_de_redis(
    armar_grafo_produccion: Callable[..., CompiledStateGraph],
) -> None:
    valores = await _estado_pausado(armar_grafo_produccion)
    assert valores["acciones_propuestas"]  # incluye los modelos de la etapa de aprobacion
    serializador = serializador_del_estado()
    recuperado = serializador.loads_typed(serializador.dumps_typed(valores))
    for campo in ("hallazgos", "analisis", "validaciones", "decisiones", "contribuciones", "acciones_propuestas", "intentos", "messages"):
        assert recuperado[campo] == valores[campo], campo
    assert type(recuperado["hallazgos"]).__name__ == "Hallazgos"
    assert type(recuperado["acciones_propuestas"][0]).__name__ == "AccionPropuesta"


async def test_sin_la_lista_declarada_los_modelos_vuelven_como_diccionarios(
    armar_grafo_produccion: Callable[..., CompiledStateGraph],
) -> None:
    """Documenta POR QUE existe `serializador_del_estado`: el serializador por defecto no los reconstruye."""
    valores = await _estado_pausado(armar_grafo_produccion)
    por_defecto = JsonPlusRedisSerializer()
    recuperado = por_defecto.loads_typed(por_defecto.dumps_typed(valores))
    assert isinstance(recuperado["hallazgos"], dict)
