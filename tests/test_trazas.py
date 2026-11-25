"""Conteo de llamadas y tokens del modelo."""

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from orquestador.trazas import ContadorDeUso


async def test_el_contador_suma_llamadas_y_tokens() -> None:
    contador = ContadorDeUso()
    mensaje = AIMessage(content="hola", usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150})
    resultado = LLMResult(generations=[[ChatGeneration(message=mensaje)]])
    await contador.on_llm_end(resultado)
    await contador.on_llm_end(resultado)
    await contador.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="sin uso"))]]))
    assert (contador.llamadas, contador.tokens_entrada, contador.tokens_salida) == (3, 240, 60)
