"""Armado del modelo de Anthropic y generacion del diagrama del grafo."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langgraph.graph.state import CompiledStateGraph

from orquestador.config import BASE_DIR

RUTA_DIAGRAMA = BASE_DIR / "docs" / "grafo.mmd"


TIMEOUT_LLM_SEGUNDOS = 60.0
REINTENTOS_LLM = 2


def crear_llm(clave: str, modelo: str, max_tokens: int) -> ChatAnthropic:
    """Modelo de Anthropic, con timeout y reintentos acotados (no esperar para siempre ni reintentar sin fin).

    No se envia `temperature`: la API de Anthropic la rechaza con este modelo
    (error 400 "temperature: Extra inputs are not permitted"), asi que queda el valor del servicio.
    """
    return ChatAnthropic(  # type: ignore[call-arg]
        model=modelo,
        api_key=clave,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        timeout=TIMEOUT_LLM_SEGUNDOS,
        max_retries=REINTENTOS_LLM,
    )


def diagrama_mermaid(grafo: CompiledStateGraph) -> str:
    """Texto Mermaid del grafo, generado por LangGraph a partir de los nodos y aristas reales."""
    return grafo.get_graph().draw_mermaid()


def _png_con_mmdc(ruta_mmd: Path, ruta_png: Path) -> bool:
    """Dibuja el PNG con el Mermaid CLI (`mmdc`) si esta instalado: no necesita red."""
    mmdc = shutil.which("mmdc")
    if mmdc is None:
        return False
    with tempfile.TemporaryDirectory() as carpeta:
        configuracion = Path(carpeta) / "puppeteer.json"
        configuracion.write_text('{"args": ["--no-sandbox"]}', encoding="utf-8")  # necesario en contenedores
        resultado = subprocess.run(
            [mmdc, "-i", str(ruta_mmd), "-o", str(ruta_png), "-p", str(configuracion), "-s", "2", "-b", "white"],
            capture_output=True, timeout=120, check=False,
        )
    return resultado.returncode == 0 and ruta_png.exists()


def guardar_diagrama(grafo: CompiledStateGraph, ruta_mmd: Path = RUTA_DIAGRAMA) -> tuple[Path, Path | None]:
    """Guarda el diagrama como texto Mermaid y como PNG, si se puede.

    El PNG se pide primero a `draw_mermaid_png()` (que usa el servicio mermaid.ink, necesita red) y, si
    falla, al Mermaid CLI local (`mmdc`). Si ninguno se puede, se guarda solo el texto y se devuelve `None`.
    """
    ruta_mmd.parent.mkdir(parents=True, exist_ok=True)
    ruta_mmd.write_text(diagrama_mermaid(grafo), encoding="utf-8")
    ruta_png = ruta_mmd.with_suffix(".png")
    try:
        ruta_png.write_bytes(grafo.get_graph().draw_mermaid_png())
        return ruta_mmd, ruta_png
    except Exception:  # noqa: BLE001  - el PNG es opcional: cualquier falla de red o de renderizado se tolera
        pass
    try:
        return (ruta_mmd, ruta_png) if _png_con_mmdc(ruta_mmd, ruta_png) else (ruta_mmd, None)
    except (OSError, subprocess.SubprocessError):
        return ruta_mmd, None
