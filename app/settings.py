"""Ajustes de la capa de produccion: todo sale de variables de entorno (archivo .env).

Los ajustes del orquestador en si (clave de Anthropic, modelos, topes de pasos) siguen en
`orquestador.config.Configuracion`; aca van solo los de la API, el worker, Redis y las trazas.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from orquestador.config import BASE_DIR, entero_del_entorno
from orquestador.errores import ErrorConfiguracion

REDIS_URL_POR_DEFECTO = "redis://localhost:6379"
PHOENIX_ENDPOINT_POR_DEFECTO = "http://localhost:6006/v1/traces"
PHOENIX_PROYECTO_POR_DEFECTO = "orquestador-gobierno-de-datos"
VERSION_PROMPTS_POR_DEFECTO = "final-v1"
JOB_TIMEOUT_POR_DEFECTO = 300
MARGEN_HUERFANOS_SEGUNDOS = 60
_VERDADERO = {"1", "true", "yes", "si", "sí", "on"}
_FALSO = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Ajustes:
    """Valores con los que se arman la API y el worker.

    - `redis_url`: Redis para la cola, el estado de los jobs y los checkpoints de LangGraph
      (tiene que ser Redis 8+ o Redis Stack: los checkpoints necesitan los modulos de busqueda y JSON).
    - `observabilidad`: si esta activa, el worker manda una traza de cada ejecucion a Phoenix.
    - `worker_concurrencia`: cuantos jobs corre a la vez un worker (cada uno es una tarea asincrona).
    - `ttl_jobs_segundos` / `ttl_checkpoints_minutos`: cuanto se conservan los datos en Redis.
    - `job_timeout_segundos`: tiempo maximo de ejecucion de un job; pasado ese tiempo queda en FAILED.
    """

    redis_url: str
    observabilidad: bool
    phoenix_endpoint: str
    phoenix_proyecto: str
    version_prompts: str
    worker_concurrencia: int
    ttl_jobs_segundos: int
    ttl_checkpoints_minutos: int
    job_timeout_segundos: int = JOB_TIMEOUT_POR_DEFECTO

    @property
    def job_huerfano_segundos(self) -> int:
        """Un job que sigue en RUNNING pasado el timeout (mas un margen) ya no lo esta corriendo nadie."""
        return self.job_timeout_segundos + MARGEN_HUERFANOS_SEGUNDOS

    @classmethod
    def desde_entorno(cls, *, cargar_dotenv: bool = True) -> "Ajustes":
        if cargar_dotenv:
            load_dotenv(BASE_DIR / ".env", override=False)
        ajustes = cls(
            redis_url=os.environ.get("REDIS_URL", "").strip() or REDIS_URL_POR_DEFECTO,
            observabilidad=_booleano("OBSERVABILIDAD", True),
            phoenix_endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "").strip() or PHOENIX_ENDPOINT_POR_DEFECTO,
            phoenix_proyecto=os.environ.get("PHOENIX_PROJECT_NAME", "").strip() or PHOENIX_PROYECTO_POR_DEFECTO,
            version_prompts=os.environ.get("VERSION_PROMPTS", "").strip() or VERSION_PROMPTS_POR_DEFECTO,
            worker_concurrencia=entero_del_entorno("WORKER_CONCURRENCIA", 5, minimo=1),
            ttl_jobs_segundos=entero_del_entorno("TTL_JOBS_SEGUNDOS", 7 * 24 * 3600, minimo=60),
            ttl_checkpoints_minutos=entero_del_entorno("TTL_CHECKPOINTS_MINUTOS", 7 * 24 * 60, minimo=1),
            job_timeout_segundos=entero_del_entorno("JOB_TIMEOUT_SEGUNDOS", JOB_TIMEOUT_POR_DEFECTO, minimo=30),
        )
        if ajustes.ttl_jobs_segundos <= ajustes.job_timeout_segundos + MARGEN_HUERFANOS_SEGUNDOS:
            raise ErrorConfiguracion(
                "TTL_JOBS_SEGUNDOS tiene que ser mayor que JOB_TIMEOUT_SEGUNDOS más "
                f"{MARGEN_HUERFANOS_SEGUNDOS} s: si no, el registro de un job expira mientras todavía corre."
            )
        if ajustes.ttl_checkpoints_minutos * 60 < ajustes.ttl_jobs_segundos:
            raise ErrorConfiguracion(
                "TTL_CHECKPOINTS_MINUTOS tiene que cubrir al menos TTL_JOBS_SEGUNDOS: "
                "si no, una tarea puede esperar aprobación sin que quede su checkpoint."
            )
        return ajustes


def _booleano(nombre: str, por_defecto: bool) -> bool:
    """Lee un booleano del entorno ("true"/"false", "1"/"0", "si"/"no")."""
    texto = os.environ.get(nombre, "").strip().lower()
    if not texto:
        return por_defecto
    if texto in _VERDADERO:
        return True
    if texto in _FALSO:
        return False
    raise ErrorConfiguracion(f"{nombre} tiene que ser true o false (vale {texto!r}).")
