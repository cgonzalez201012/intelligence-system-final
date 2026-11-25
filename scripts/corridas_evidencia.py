"""Corridas de evidencia: 6 tareas distintas contra el sistema REAL, con su resultado y la guía para ver sus trazas.

Necesita el sistema levantado (`./run.sh`). Cada corrida cubre un camino distinto del sistema:

  1. solo políticas (RAG)          -> responde citando documentos recuperados
  2. incidentes + políticas        -> se pausa; el script APRUEBA y se ejecutan los escalamientos
  3. incidentes críticos           -> se pausa; el script RECHAZA y no se ejecuta nada
  4. incidentes sin críticos       -> termina sin pedir aprobación
  5. glosario (RAG, otra categoría)-> responde citando el glosario
  6. fuera de alcance              -> el investigador lo declara y no inventa nada

Deja docs/evidencia/corridas.json (datos completos) y docs/evidencia/corridas.md (tabla para el README) y marca
cada corrida con OK / REVISAR según lo que se esperaba. Las trazas se ven en Phoenix (http://localhost:6006):
cada job es una "session" con su job_id.

Uso:  python scripts/corridas_evidencia.py            (o: make evidencia)
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "docs" / "evidencia"
FINALES = {"DONE", "FAILED"}
Decision = Literal["aprobar", "rechazar"] | None


@dataclass(frozen=True)
class Caso:
    nombre: str
    tarea: str
    decision: Decision = None        # qué hace el script si el job queda pausado
    espera_pausa: bool = False       # ¿se espera que pida aprobación?
    espera_fuentes: bool = False     # ¿se espera que cite políticas del RAG?
    espera_fuera_de_alcance: bool = False


CASOS: tuple[Caso, ...] = (
    Caso("1. Solo políticas (RAG)", "¿Cuál es el plazo para resolver un incidente crítico y a quién se escala si se vence?", espera_fuentes=True),
    Caso("2. Incidentes + políticas, APROBADO", "Priorizá los incidentes críticos abiertos del maestro de artículos y citá la política que fija sus plazos.", decision="aprobar", espera_pausa=True, espera_fuentes=True),
    Caso("3. Incidentes críticos, RECHAZADO", "Priorizá los incidentes críticos abiertos del maestro de proveedores.", decision="rechazar", espera_pausa=True),
    Caso("4. Sin críticos (sin aprobación)", "Priorizá los incidentes abiertos del maestro de sucursales."),
    Caso("5. Glosario (RAG)", "¿Qué es un Data Steward según el glosario de Gobierno de Datos y qué responsabilidades tiene?", espera_fuentes=True),
    Caso("6. Fuera de alcance", "¿Cuál es la capital de Francia?", espera_fuera_de_alcance=True),
)


@dataclass
class Corrida:
    caso: str
    tarea: str
    job_id: str = ""
    estado: str = ""
    estado_final: str = ""
    paso_por_aprobacion: bool = False
    decision: str = ""
    agentes: list[str] = field(default_factory=list)
    fuentes_citadas: list[str] = field(default_factory=list)
    acciones_ejecutadas: int = 0
    segundos: float = 0.0
    tokens_entrada: int = 0
    tokens_salida: int = 0
    respuesta: str = ""
    error: str = ""
    veredicto: str = ""
    observaciones: list[str] = field(default_factory=list)


def evaluar(caso: Caso, c: Corrida) -> tuple[str, list[str]]:
    """Compara lo ocurrido con lo esperado: devuelve ('OK' | 'REVISAR', motivos)."""
    problemas: list[str] = []
    if c.estado != "DONE":
        problemas.append(f"terminó en {c.estado or 'sin estado'}: {c.error or 'sin detalle'}")
    if caso.espera_pausa != c.paso_por_aprobacion:
        problemas.append("se esperaba pausa por aprobación" if caso.espera_pausa else "no se esperaba pausa por aprobación")
    if caso.espera_fuentes and not c.fuentes_citadas:
        problemas.append("se esperaba que citara políticas del RAG y no citó ninguna")
    if caso.decision == "aprobar" and c.acciones_ejecutadas == 0:
        problemas.append("se aprobó pero no se ejecutó ninguna acción")
    if caso.decision == "rechazar" and c.acciones_ejecutadas != 0:
        problemas.append("se rechazó pero se ejecutaron acciones")
    if caso.espera_fuera_de_alcance and "fuera" not in c.respuesta.lower() and "no puedo" not in c.respuesta.lower() and "no trata" not in c.respuesta.lower():
        problemas.append("se esperaba que declarara la consulta fuera de alcance")
    return ("OK" if not problemas else "REVISAR"), problemas


async def _esperar(cliente: httpx.AsyncClient, job_id: str, hasta: set[str], timeout_s: float) -> dict[str, Any]:
    inicio = time.monotonic()
    while True:
        job = (await cliente.get(f"/tasks/{job_id}")).raise_for_status().json()
        if job["estado"] in hasta:
            return job
        if time.monotonic() - inicio > timeout_s:
            raise TimeoutError(f"el job {job_id} siguió en {job['estado']} pasados {timeout_s:.0f} s")
        await asyncio.sleep(1.0)


async def correr_caso(cliente: httpx.AsyncClient, caso: Caso, *, timeout_s: float = 300.0) -> Corrida:
    resultado = Corrida(caso=caso.nombre, tarea=caso.tarea)
    inicio = time.monotonic()
    try:
        creado = (await cliente.post("/tasks", json={"tarea": caso.tarea})).raise_for_status().json()
        resultado.job_id = creado["job_id"]
        job = await _esperar(cliente, resultado.job_id, FINALES | {"WAITING_APPROVAL"}, timeout_s)
        if job["estado"] == "WAITING_APPROVAL":
            resultado.paso_por_aprobacion = True
            decision = caso.decision or "rechazar"
            cuerpo = {"aprobada": decision == "aprobar", "revisor": "evidencia-script", "comentario": f"Decisión automática de la corrida ({decision})."}
            (await cliente.post(f"/tasks/{resultado.job_id}/approve", json=cuerpo)).raise_for_status()
            resultado.decision = decision
            job = await _esperar(cliente, resultado.job_id, FINALES, timeout_s)
        resultado.estado = job["estado"]
        resultado.error = job.get("error") or ""
        resultado.segundos = job.get("segundos_de_ejecucion", 0.0)
        resultado.tokens_entrada = job["uso"]["tokens_entrada"]
        resultado.tokens_salida = job["uso"]["tokens_salida"]
        if job.get("resultado"):
            r = job["resultado"]
            resultado.estado_final = r["estado_final"]
            resultado.respuesta = r["respuesta"]
            resultado.agentes = r["agentes_que_intervinieron"]
            resultado.fuentes_citadas = r.get("fuentes_citadas", [])
            resultado.acciones_ejecutadas = sum(1 for a in r["acciones"]["ejecutadas"] if a["ejecutada"])
    except (httpx.HTTPError, TimeoutError, KeyError) as error:
        resultado.estado = resultado.estado or "ERROR_DEL_SCRIPT"
        resultado.error = f"{type(error).__name__}: {error}"
    resultado.segundos = resultado.segundos or round(time.monotonic() - inicio, 1)
    resultado.veredicto, resultado.observaciones = evaluar(caso, resultado)
    return resultado


def tabla_markdown(corridas: list[Corrida]) -> str:
    filas = [
        "| Corrida | job_id | Estado | Agentes | Fuentes del RAG citadas | Acciones | Seg. | Tokens (ent/sal) | Veredicto |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in corridas:
        fuentes = "<br>".join(c.fuentes_citadas) or "—"
        aprobacion = f" ({c.decision})" if c.decision else ""
        filas.append(
            f"| {c.caso} | `{c.job_id[:8]}…` | {c.estado_final or c.estado}{aprobacion} | {', '.join(c.agentes) or '—'} | {fuentes} "
            f"| {c.acciones_ejecutadas} | {c.segundos:.1f} | {c.tokens_entrada}/{c.tokens_salida} | {c.veredicto} |"
        )
    return "\n".join(filas) + "\n"


async def main_async(url: str, concurrencia: int) -> int:
    limite = asyncio.Semaphore(concurrencia)
    async with httpx.AsyncClient(base_url=url, timeout=30.0) as cliente:
        try:
            (await cliente.get("/health")).raise_for_status()
        except httpx.HTTPError as error:
            print(f"No pude hablar con la API en {url} ({error}). ¿Corriste ./run.sh?", file=sys.stderr)
            return 2

        async def una(caso: Caso) -> Corrida:
            async with limite:
                print(f"→ {caso.nombre}")
                return await correr_caso(cliente, caso)

        corridas = list(await asyncio.gather(*(una(c) for c in CASOS)))

    CARPETA.mkdir(parents=True, exist_ok=True)
    (CARPETA / "corridas.json").write_text(
        json.dumps({"generado_en": datetime.now().isoformat(timespec="seconds"), "url": url, "corridas": [asdict(c) for c in corridas]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (CARPETA / "corridas.md").write_text(tabla_markdown(corridas), encoding="utf-8")
    print()
    for c in corridas:
        marca = "✔" if c.veredicto == "OK" else "✘"
        print(f"{marca} {c.caso}: {c.estado_final or c.estado} · job {c.job_id}")
        for motivo in c.observaciones:
            print(f"    - {motivo}")
    print("\nGuardado en docs/evidencia/corridas.json y corridas.md")
    print("Trazas: http://localhost:6006 → proyecto del orquestador → pestaña Sessions (una por job_id de arriba).")
    return 0 if all(c.veredicto == "OK" for c in corridas) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrencia", type=int, default=3, help="corridas a la vez (por defecto 3: cuida la cuota de Gemini)")
    argumentos = parser.parse_args()
    sys.exit(asyncio.run(main_async(argumentos.url, argumentos.concurrencia)))


if __name__ == "__main__":
    main()
