"""Conteo del uso del modelo (llamadas y tokens) de una ejecucion, para dejarlo en el registro del job."""

from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult


class ContadorDeUso(AsyncCallbackHandler):
    """Cuenta las llamadas al modelo y los tokens de entrada y salida de TODA la corrida.

    Se pasa en `config["callbacks"]` al invocar el grafo y llega a todos los nodos, incluidos
    los especialistas (cada reintento o ronda de herramientas se paga y se cuenta).
    """

    def __init__(self) -> None:
        self.llamadas = 0
        self.tokens_entrada = 0
        self.tokens_salida = 0

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.llamadas += 1
        for generaciones in response.generations:
            for generacion in generaciones:
                uso = getattr(getattr(generacion, "message", None), "usage_metadata", None)
                if uso:
                    self.tokens_entrada += int(uso.get("input_tokens", 0))
                    self.tokens_salida += int(uso.get("output_tokens", 0))
