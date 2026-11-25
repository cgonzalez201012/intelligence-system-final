"""Comprueba, contra el Redis REAL, que todo lo que la API necesita funciona. No gasta la API de Anthropic.

Verifica, en orden:
  1. Que Redis responde y que tiene los modulos de busqueda y JSON (los pide el checkpointer).
  2. Que un job con un incidente critico se PAUSA esperando aprobacion y que el estado queda en Redis.
  3. Que, con otra conexion y otro checkpointer (como si fuera otro worker, o el mismo despues de un reinicio),
     el job se REANUDA con la decision, y el estado vuelve completo y con sus modelos (no como diccionarios).
  4. Que la accion aprobada se ejecuta una sola vez (aviso en la bandeja de salida).

Usa los modelos falsos de las pruebas, asi que se puede correr las veces que haga falta sin costo.
Uso:  python scripts/verificar_entorno.py        (o: make verificar)
"""

import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from langgraph.types import Command  # noqa: E402
from redis.asyncio import Redis  # noqa: E402
from redis.exceptions import RedisError  # noqa: E402

from app.graph import abrir_checkpointer, construir_grafo_produccion, verificar_modulos_de_redis  # noqa: E402
from app.notificador import NotificadorRedis  # noqa: E402
from app.settings import Ajustes  # noqa: E402
from orquestador.config import Configuracion  # noqa: E402
from orquestador.errores import ErrorConfiguracion  # noqa: E402
from orquestador.state import Hallazgos, estado_inicial  # noqa: E402
from tests.fakes import (  # noqa: E402
    SupervisorFalso,
    decidir,
    especialistas_falsos,
    guion_analista,
    guion_investigador,
    informe,
)

BANDEJA_DE_PRUEBA = "verificacion:escalaciones"


def _ok(texto: str) -> None:
    print(f"  [ok] {texto}")


def _grafo(checkpointer: Any, notificador: NotificadorRedis, config: Configuracion) -> CompiledStateGraph:
    supervisor = SupervisorFalso(
        guion=[decidir("investigador", "r"), decidir("analista", "c"), decidir("redactor", "e"), decidir("FINISH")]
    )
    especialistas = especialistas_falsos(guion_investigador(), guion_analista(), [informe("INC-1001", "INC-1002")])
    return construir_grafo_produccion(supervisor, especialistas, config, checkpointer=checkpointer, notificador=notificador)


async def verificar(redis_url: str) -> None:
    config = Configuracion("k", "m", "m", 10, 3, 12)
    job_id = f"verificacion-{uuid.uuid4().hex[:8]}"
    hilo: Any = {"configurable": {"thread_id": job_id}, "recursion_limit": config.recursion_limit_grafo}
    redis = Redis.from_url(redis_url, decode_responses=True)
    try:
        print(f"1. Redis en {redis_url}")
        try:
            await redis.ping()
        except Exception as error:  # noqa: BLE001
            raise ErrorConfiguracion(
                f"No puedo conectarme a Redis ({type(error).__name__}). ¿Levantaste `docker compose up -d redis`?"
            ) from error
        version = (await redis.info("server")).get("redis_version")
        _ok(f"responde (versión {version})")
        await verificar_modulos_de_redis(redis)
        _ok("tiene los módulos de búsqueda y JSON")

        notificador = NotificadorRedis(redis, ttl_segundos=300, bandeja=BANDEJA_DE_PRUEBA)

        print("2. El job crítico se pausa y el estado queda en Redis")
        async with abrir_checkpointer(redis_url, ttl_minutos=5) as checkpointer:
            grafo = _grafo(checkpointer, notificador, config)
            await grafo.ainvoke(estado_inicial("Priorizá los incidentes críticos de artículos."), hilo)
            foto = await grafo.aget_state(hilo)
            assert foto.next == ("aprobacion_humana",), f"se esperaba una pausa en aprobacion_humana, quedó en {foto.next}"
            assert foto.interrupts, "el grafo no dejó la interrupción registrada"
            _ok(f"pausado en '{foto.next[0]}' con {len(foto.values['acciones_propuestas'])} acciones por aprobar")
            assert not await redis.llen(BANDEJA_DE_PRUEBA), "se ejecutó una acción SIN aprobación"
            _ok("todavía no se ejecutó nada (no hay avisos enviados)")

        print("3. Otra conexión (otro worker o un reinicio) retoma el job")
        async with abrir_checkpointer(redis_url, ttl_minutos=5) as otro_checkpointer:
            otro_grafo = _grafo(otro_checkpointer, notificador, config)
            foto = await otro_grafo.aget_state(hilo)
            assert foto.next == ("aprobacion_humana",), "el estado no sobrevivió al cambio de conexión"
            assert isinstance(foto.values["hallazgos"], Hallazgos), "el estado volvió como diccionarios, no como modelos"
            _ok("el estado se recuperó completo y con sus modelos")
            decision = {"aprobada": True, "revisor": "verificar_entorno", "comentario": "prueba automática"}
            final = await otro_grafo.ainvoke(Command(resume=decision), hilo)
            ejecutadas = final["acciones_ejecutadas"]
            assert ejecutadas and all(a.ejecutada for a in ejecutadas), "no se ejecutaron las acciones aprobadas"
            _ok(f"reanudó con la decisión y ejecutó {len(ejecutadas)} acciones")

        print("4. La acción se ejecutó una sola vez")
        avisos = await redis.llen(BANDEJA_DE_PRUEBA)
        assert avisos == len(ejecutadas), f"hay {avisos} avisos para {len(ejecutadas)} acciones"
        _ok(f"{avisos} avisos en la bandeja de salida, uno por incidente")
    finally:
        try:  # limpieza de lo que dejo la prueba; si Redis no responde no hay nada que limpiar
            claves = [BANDEJA_DE_PRUEBA, *[clave async for clave in redis.scan_iter(f"escalacion:{job_id}:*")]]
            await redis.delete(*claves)
        except RedisError:
            pass
        await redis.aclose()
    print("\nTodo bien: el Redis de este entorno sirve para la API.")


def main() -> int:
    ajustes = Ajustes.desde_entorno()
    try:
        asyncio.run(verificar(ajustes.redis_url))
    except ErrorConfiguracion as error:
        print(f"  [x] {error}", file=sys.stderr)
        return 1
    except AssertionError as error:
        print(f"  [x] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
