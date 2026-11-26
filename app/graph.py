"""El orquestador multi-agente + la etapa de aprobación humana + el checkpointer de Redis.

El nucleo (supervisor, investigador, analista, redactor, validacion) vive en `orquestador/` tal
como lo definí. Acá solo se arma la versión de producción:

    START -> supervisor <-> (investigador | analista | redactor) -> validacion -> supervisor
                 |
                 +-- FINISH -> proponer_acciones -> aprobacion_humana -> ejecutar_acciones -> END

y se compila con un checkpointer de Redis, de modo que el estado de cada hilo (un hilo = un job)
sobrevive a un reinicio y puede retomarlo cualquier worker.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer
from langgraph.graph import END
from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis

from app.hitl import (
    NODO_APROBACION,
    NODO_EJECUTAR,
    NODO_PROPONER,
    crear_nodo_ejecutar,
    decidir_tras_aprobar,
    decidir_tras_proponer,
    nodo_aprobacion_humana,
    nodo_proponer_acciones,
)
from app.notificador import Notificador
from orquestador.config import Configuracion
from orquestador.ejecucion import crear_llm
from orquestador.errores import ErrorConfiguracion
from orquestador.graph import armar_grafo
from orquestador.state import CLASES_DEL_ESTADO

MODULOS_REQUERIDOS = {"search": "RediSearch (módulo de búsqueda)", "ReJSON": "RedisJSON"}
_ALIAS_DE_MODULOS = {"searchlight": "search", "rejson": "ReJSON"}


def construir_grafo_produccion(
    llm_supervisor: BaseChatModel,
    llm_especialistas: BaseChatModel,
    config: Configuracion,
    *,
    checkpointer: BaseCheckpointSaver[Any],
    notificador: Notificador,
) -> CompiledStateGraph:
    """Arma y compila el grafo de produccion con los modelos y el checkpointer que se le pasen."""
    grafo = armar_grafo(
        llm_supervisor,
        llm_especialistas,
        max_pasos=config.max_pasos,
        max_intentos=config.max_intentos_por_agente,
        recursion_especialista=config.recursion_especialista,
        destino_cierre=NODO_PROPONER,
    )
    nuevos: dict[str, Any] = {
        NODO_PROPONER: nodo_proponer_acciones,
        NODO_APROBACION: nodo_aprobacion_humana,
        NODO_EJECUTAR: crear_nodo_ejecutar(notificador),
    }
    for nombre, nodo in nuevos.items():
        grafo.add_node(nombre, nodo)
    grafo.add_conditional_edges(NODO_PROPONER, decidir_tras_proponer, {NODO_APROBACION: NODO_APROBACION, END: END})
    grafo.add_conditional_edges(NODO_APROBACION, decidir_tras_aprobar, {NODO_EJECUTAR: NODO_EJECUTAR, END: END})
    grafo.add_edge(NODO_EJECUTAR, END)
    return grafo.compile(name="orquestador-produccion", checkpointer=checkpointer)


def crear_grafo_con_modelos_reales(
    config: Configuracion, *, checkpointer: BaseCheckpointSaver[Any], notificador: Notificador
) -> CompiledStateGraph:
    """Igual que `construir_grafo_produccion`, con los modelos de Anthropic de la configuracion."""
    supervisor = crear_llm(config.anthropic_api_key, config.modelo_supervisor, max_tokens=1024)
    especialistas = crear_llm(config.anthropic_api_key, config.modelo_especialistas, max_tokens=2048)
    return construir_grafo_produccion(
        supervisor, especialistas, config, checkpointer=checkpointer, notificador=notificador
    )


# ---------- Checkpointer de Redis ----------


def serializador_del_estado() -> JsonPlusRedisSerializer:
    """Serializador de checkpoints que SI reconstruye los modelos Pydantic del estado.

    Por seguridad, LangGraph solo revive las clases que se le declaran. Sin esta lista, un job
    retomado desde Redis (por ejemplo, por otro worker al aprobar) recibiria diccionarios en lugar
    de `Hallazgos`, `Analisis`, etc., y los nodos fallarian.
    """
    permitidas = [(*clase.__module__.split("."), clase.__name__) for clase in CLASES_DEL_ESTADO]
    return JsonPlusRedisSerializer(allowed_json_modules=permitidas, allowed_msgpack_modules=permitidas)


async def verificar_modulos_de_redis(redis: Redis) -> None:
    """Falla temprano y con un mensaje claro si Redis no tiene los modulos que necesita el checkpointer.

    Redis 8+ los trae incluidos; las versiones anteriores necesitan Redis Stack. Sin ellos, el error
    recien apareceria al guardar el primer checkpoint, lejos de la causa.
    """
    try:
        modulos = await redis.module_list()
    except Exception as error:  # noqa: BLE001 - algunos Redis administrados no permiten MODULE LIST
        raise ErrorConfiguracion(
            "No pude consultar los módulos de Redis (MODULE LIST). Usá Redis 8+ o Redis Stack "
            f"(ver README): {type(error).__name__}"
        ) from error
    presentes: set[str] = set()
    for modulo in modulos:
        nombre = modulo.get("name", modulo.get(b"name", b""))
        nombre = nombre.decode() if isinstance(nombre, bytes) else str(nombre)
        presentes.add(_ALIAS_DE_MODULOS.get(nombre.lower(), nombre))
    faltan = [descripcion for clave, descripcion in MODULOS_REQUERIDOS.items() if clave not in presentes]
    if faltan:
        raise ErrorConfiguracion(
            "Este Redis no tiene los módulos que necesita el checkpointer de LangGraph: "
            + ", ".join(faltan)
            + ". Usá Redis 8+ o Redis Stack (docker compose up redis lo levanta bien; ver README)."
        )


@asynccontextmanager
async def abrir_checkpointer(redis_url: str, *, ttl_minutos: int) -> AsyncIterator[AsyncRedisSaver]:
    """Abre el checkpointer de Redis con el serializador del estado, crea sus indices y lo entrega listo."""
    ttl = {"default_ttl": ttl_minutos, "refresh_on_read": True}
    async with AsyncRedisSaver.from_conn_string(redis_url, ttl=ttl) as checkpointer:
        checkpointer.serde = serializador_del_estado()
        await checkpointer.asetup()
        yield checkpointer
