"""Observabilidad: una traza de cada ejecucion, con un span por nodo, por llamada al modelo y por herramienta.

Se usa Arize Phoenix con el estandar OpenInference. El instrumentador de LangChain se engancha a
los callbacks de LangChain/LangGraph, asi que no hace falta decorar ninguna funcion a mano:
cada nodo del grafo, cada llamada a Claude (con sus tokens de entrada y salida) y cada herramienta
aparecen solos como spans de la traza de la tarea. Con los tokens y el nombre del modelo, Phoenix
calcula el costo de cada ejecucion.

Cada job se etiqueta con `session_id = job_id` (en Phoenix, la pestana Sessions agrupa por job)
y con la version de los prompts, para poder relacionar un cambio de comportamiento con un cambio de prompt.
"""

import logging
import os
from contextlib import AbstractContextManager
from typing import Any

from openinference.instrumentation import using_attributes
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import TracerProvider

from app.settings import Ajustes

logger = logging.getLogger(__name__)


def instrumentar_langchain(proveedor: TracerProvider) -> None:
    """Engancha el instrumentador de LangChain/LangGraph al proveedor de trazas que se le pase."""
    LangChainInstrumentor().instrument(tracer_provider=proveedor, skip_dep_check=True)


def silenciar_trazas_propias_del_sdk() -> None:
    """Apaga los spans que el SDK de Anthropic (2.x) emite por su cuenta cuando hay un proveedor de trazas global.

    Sin esto, cada llamada al modelo deja DOS spans: el de LangChain (hijo del nodo, con tokens y costo)
    y otro `anthropic.messages.create` que queda como traza suelta, sin padre. Esas trazas sueltas duplican
    los conteos y falsean las metricas del proyecto (p50/p95/p99 mezclarian llamadas de 2 s con jobs de 40 s).
    Hay que llamarla antes de crear los clientes del modelo, porque cada cliente lee la variable al construirse.
    """
    os.environ["ANTHROPIC_OPEN_TELEMETRY"] = "false"


def iniciar_observabilidad(ajustes: Ajustes) -> bool:
    """Conecta el proceso con Phoenix. Devuelve True si quedo activa.

    Si Phoenix no esta disponible, el worker sigue funcionando sin trazas (se avisa en el log): una
    herramienta de monitoreo caida no debe tirar abajo el servicio que monitorea.
    """
    if not ajustes.observabilidad:
        logger.info("Observabilidad desactivada (OBSERVABILIDAD=false).")
        return False
    silenciar_trazas_propias_del_sdk()
    try:
        from phoenix.otel import register

        proveedor = register(
            project_name=ajustes.phoenix_proyecto,
            endpoint=ajustes.phoenix_endpoint,
            batch=True,
            set_global_tracer_provider=True,
            auto_instrument=False,
            verbose=False,
        )
        instrumentar_langchain(proveedor)
    except Exception:  # noqa: BLE001 - ver docstring: el monitoreo no debe frenar el servicio
        logger.exception("No pude iniciar la observabilidad: sigo sin trazas.")
        return False
    logger.info("Trazas hacia %s (proyecto %s).", ajustes.phoenix_endpoint, ajustes.phoenix_proyecto)
    return True


def etiquetas_de_traza(job_id: str, version_prompts: str, accion: str = "ejecutar") -> AbstractContextManager[Any]:
    """Atributos que reciben TODOS los spans creados dentro del bloque `with`."""
    return using_attributes(
        session_id=job_id,
        metadata={"job_id": job_id, "accion": accion, "version_prompts": version_prompts},
        tags=[f"prompts:{version_prompts}", accion],
        prompt_template_version=version_prompts,
    )
