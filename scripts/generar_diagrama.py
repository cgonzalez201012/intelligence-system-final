"""Genera docs/grafo.mmd (y docs/grafo.png si hay red) a partir del grafo de produccion real.

No llama a ningun modelo ni a Redis: el diagrama sale de los nodos y aristas que declara LangGraph.
Uso:  python scripts/generar_diagrama.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from app.graph import construir_grafo_produccion  # noqa: E402
from orquestador.config import BASE_DIR, Configuracion  # noqa: E402
from orquestador.ejecucion import crear_llm, guardar_diagrama  # noqa: E402
from orquestador.state import AccionPropuesta, ResultadoAccion  # noqa: E402


class _NotificadorNulo:
    async def notificar(self, job_id: str, accion: AccionPropuesta) -> ResultadoAccion:
        raise NotImplementedError("solo se usa para dibujar el grafo")


def main() -> None:
    config = Configuracion.desde_entorno(requerir_clave=False)
    llm = crear_llm("clave-de-mentira", config.modelo_supervisor, max_tokens=16)  # no se invoca nunca
    grafo = construir_grafo_produccion(
        llm, llm, config, checkpointer=InMemorySaver(), notificador=_NotificadorNulo()
    )
    ruta_mmd, ruta_png = guardar_diagrama(grafo, BASE_DIR / "docs" / "grafo.mmd")
    print(f"Diagrama de texto: {ruta_mmd.relative_to(BASE_DIR)}")
    print(f"Diagrama PNG: {ruta_png.relative_to(BASE_DIR) if ruta_png else 'no se pudo generar (sin red); el texto alcanza'}")


if __name__ == "__main__":
    main()
