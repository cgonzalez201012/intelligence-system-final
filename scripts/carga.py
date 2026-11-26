"""Prueba de carga: N peticiones concurrentes contra la API y resumen de latencias.

Necesita la API y al menos un worker corriendo (ver README). Dispara las N tareas AL MISMO TIEMPO,
consulta cada job hasta que termina y calcula la latencia de cada uno (p50 y p95) y el uso de tokens.
El costo en dolares por ejecucion y el p95 "oficial" se leen en el dashboard de Phoenix; este script
deja el resumen en docs/evidencia/ para poder contrastarlo con lo que muestra el dashboard.

Uso:
    python scripts/carga.py                      # 5 tareas que terminan solas (maestro de sucursales)
    python scripts/carga.py --perfil con-aprobacion   # 5 tareas con incidentes criticos; las aprueba el script
    python scripts/carga.py --n 10 --url http://localhost:8000
"""

import argparse
import asyncio
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_EVIDENCIA = RAIZ / "docs" / "evidencia"
ESTADOS_FINALES = {"DONE", "FAILED"}

TAREAS_POR_PERFIL: dict[str, list[str]] = {
    # Sucursales no tiene incidentes criticos: el grafo termina sin pedir aprobacion.
    "sin-aprobacion": [
        "Priorizá los incidentes abiertos del maestro de sucursales.",
        "¿Qué incidentes de sucursales superaron su SLA al cierre de octubre?",
        "Armá un resumen de los incidentes de sucursales por severidad y responsable.",
        "Listá los incidentes en curso del maestro de sucursales y decí cuál conviene atender primero.",
        "Redactá un informe corto sobre el estado de calidad de datos del maestro de sucursales.",
    ],
    # Artículos y proveedores tienen críticos vencidos: el grafo se pausa y pide aprobación.
    "con-aprobacion": [
        "Priorizá los incidentes críticos abiertos del maestro de artículos.",
        "Priorizá los incidentes críticos abiertos del maestro de proveedores.",
        "¿Qué incidentes críticos de artículos superaron su SLA al cierre de octubre?",
        "Armá un resumen de los incidentes críticos de proveedores y su responsable.",
        "Redactá un informe corto sobre los incidentes críticos del maestro de artículos.",
    ],
}


@dataclass
class ResultadoJob:
    """Lo que se midio de un job."""

    indice: int
    job_id: str = ""
    estado_final: str = ""
    segundos_hasta_aceptado: float = 0.0   # lo que tarda la API en responder 202 (no espera al modelo)
    segundos_totales: float = 0.0          # desde que se envia hasta que el job termina
    segundos_de_ejecucion: float = 0.0     # lo que midio el worker (sin la espera en la cola)
    tokens_entrada: int = 0
    tokens_salida: int = 0
    llamadas_al_modelo: int = 0
    aprobado_por_el_script: bool = False
    error: str | None = None


@dataclass
class Resumen:
    """Resumen de toda la prueba."""

    inicio: str
    url: str
    perfil: str
    peticiones: int
    exitosas: int
    fallidas: int
    segundos_de_pared: float
    latencia_p50_s: float | None
    latencia_p95_s: float | None
    latencia_maxima_s: float | None
    aceptacion_maxima_s: float | None
    tokens_entrada_total: int
    tokens_salida_total: int
    tokens_por_ejecucion_promedio: float | None
    costo_por_ejecucion_usd: float | None
    detalle: list[ResultadoJob] = field(default_factory=list)


def percentil(valores: list[float], p: float) -> float | None:
    """Percentil por el metodo del rango mas cercano (nearest-rank): sin interpolar, usa un valor real medido.

    Con 5 muestras, p95 es el valor mas alto (ceil(0.95 * 5) = 5): es lo esperable con tan pocos datos.
    """
    if not valores:
        return None
    ordenados = sorted(valores)
    posicion = max(1, math.ceil(p / 100 * len(ordenados)))
    return ordenados[posicion - 1]


def costo_usd(tokens_entrada: int, tokens_salida: int, precio_entrada: float | None, precio_salida: float | None) -> float | None:
    """Costo en dolares a partir de los precios por millon de tokens; None si no se informaron los precios."""
    if precio_entrada is None or precio_salida is None:
        return None
    return (tokens_entrada * precio_entrada + tokens_salida * precio_salida) / 1_000_000


def resumir(
    resultados: list[ResultadoJob], *, url: str, perfil: str, inicio: str, segundos_de_pared: float,
    precio_entrada: float | None = None, precio_salida: float | None = None,
) -> Resumen:
    """Arma el resumen de la prueba a partir de lo medido en cada job."""
    exitosas = [r for r in resultados if r.estado_final == "DONE"]
    latencias = [r.segundos_totales for r in exitosas]
    entrada = sum(r.tokens_entrada for r in exitosas)
    salida = sum(r.tokens_salida for r in exitosas)
    por_ejecucion = costo_usd(entrada, salida, precio_entrada, precio_salida)
    return Resumen(
        inicio=inicio, url=url, perfil=perfil, peticiones=len(resultados),
        exitosas=len(exitosas), fallidas=len(resultados) - len(exitosas),
        segundos_de_pared=round(segundos_de_pared, 2),
        latencia_p50_s=_redondear(percentil(latencias, 50)),
        latencia_p95_s=_redondear(percentil(latencias, 95)),
        latencia_maxima_s=_redondear(max(latencias, default=None)),
        aceptacion_maxima_s=_redondear(max((r.segundos_hasta_aceptado for r in resultados), default=None)),
        tokens_entrada_total=entrada, tokens_salida_total=salida,
        tokens_por_ejecucion_promedio=_redondear((entrada + salida) / len(exitosas)) if exitosas else None,
        costo_por_ejecucion_usd=None if por_ejecucion is None or not exitosas else round(por_ejecucion / len(exitosas), 6),
        detalle=resultados,
    )


def _redondear(valor: float | None) -> float | None:
    return None if valor is None else round(valor, 2)


async def correr_peticion(
    cliente: httpx.AsyncClient, indice: int, tarea: str, *, aprobar: bool, tiempo_maximo: float, cada: float
) -> ResultadoJob:
    """Envia una tarea y la sigue hasta que termina (aprobandola si hace falta y se pidio)."""
    resultado = ResultadoJob(indice=indice)
    inicio = time.perf_counter()
    try:
        respuesta = await cliente.post("/tasks", json={"tarea": tarea})
        resultado.segundos_hasta_aceptado = time.perf_counter() - inicio
        if respuesta.status_code != 202:
            resultado.estado_final, resultado.error = "FAILED", f"POST /tasks devolvio {respuesta.status_code}: {respuesta.text[:200]}"
            return resultado
        resultado.job_id = respuesta.json()["job_id"]
        estado: dict[str, Any] = {}
        while time.perf_counter() - inicio < tiempo_maximo:
            consulta = await cliente.get(f"/tasks/{resultado.job_id}")
            if consulta.status_code != 200:
                resultado.error = f"GET /tasks/{{id}} devolvio {consulta.status_code}: {consulta.text[:200]}"
                break
            estado = consulta.json()
            if estado["estado"] in ESTADOS_FINALES:
                break
            if estado["estado"] == "WAITING_APPROVAL":
                if not aprobar:
                    break
                decision = {"aprobada": True, "revisor": "prueba-de-carga", "comentario": "Aprobada automaticamente por scripts/carga.py"}
                aprobacion = await cliente.post(f"/tasks/{resultado.job_id}/approve", json=decision)
                if aprobacion.status_code not in (200, 202):
                    resultado.error = f"/approve devolvio {aprobacion.status_code}: {aprobacion.text[:200]}"
                    break
                resultado.aprobado_por_el_script = True
            await asyncio.sleep(cada)
        resultado.segundos_totales = time.perf_counter() - inicio
        resultado.estado_final = estado.get("estado", "SIN_RESPUESTA")
        if resultado.estado_final not in ESTADOS_FINALES and resultado.error is None:
            resultado.error = f"no termino: quedo en {resultado.estado_final}"
            resultado.estado_final = "FAILED" if resultado.estado_final != "WAITING_APPROVAL" else "WAITING_APPROVAL"
        resultado.segundos_de_ejecucion = float(estado.get("segundos_de_ejecucion", 0.0))
        uso = estado.get("uso") or {}
        resultado.tokens_entrada = int(uso.get("tokens_entrada", 0))
        resultado.tokens_salida = int(uso.get("tokens_salida", 0))
        resultado.llamadas_al_modelo = int(uso.get("llamadas_al_modelo", 0))
        if estado.get("error"):
            resultado.error = str(estado["error"])
    except (httpx.HTTPError, ValueError, KeyError) as error:  # red caida, respuesta que no es JSON o sin "estado"
        resultado.estado_final, resultado.error = "FAILED", f"{type(error).__name__}: {error}"
        resultado.segundos_totales = time.perf_counter() - inicio
    return resultado


async def correr_prueba(
    url: str, tareas: list[str], *, aprobar: bool, tiempo_maximo: float, cada: float = 1.0,
    transporte: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[ResultadoJob], float]:
    """Dispara todas las tareas a la vez y espera a que terminen. Devuelve los resultados y el tiempo total.

    `transporte` solo lo usan las pruebas (para hablar con la API en memoria, sin red).
    """
    limites = httpx.Limits(max_connections=len(tareas) * 2)
    async with httpx.AsyncClient(base_url=url, timeout=30.0, limits=limites, transport=transporte) as cliente:
        inicio = time.perf_counter()
        resultados = await asyncio.gather(
            *(
                correr_peticion(cliente, i, tarea, aprobar=aprobar, tiempo_maximo=tiempo_maximo, cada=cada)
                for i, tarea in enumerate(tareas, start=1)
            )
        )
        return list(resultados), time.perf_counter() - inicio


def imprimir(resumen: Resumen) -> None:
    """Tabla legible en la terminal."""
    print(f"\n{'#':>2}  {'job_id':<34} {'estado':<17} {'aceptado':>9} {'total':>8} {'worker':>8} {'tokens(in/out)':>16}")
    for r in resumen.detalle:
        print(
            f"{r.indice:>2}  {r.job_id:<34} {r.estado_final:<17} {r.segundos_hasta_aceptado:>8.3f}s "
            f"{r.segundos_totales:>7.1f}s {r.segundos_de_ejecucion:>7.1f}s {r.tokens_entrada:>7}/{r.tokens_salida:<8}"
        )
        if r.error:
            print(f"    -> {r.error}")
    print(f"\nPeticiones: {resumen.peticiones}  exitosas: {resumen.exitosas}  fallidas: {resumen.fallidas}")
    print(f"Tiempo total de la prueba: {resumen.segundos_de_pared} s")
    print(f"Latencia p50: {resumen.latencia_p50_s} s | p95: {resumen.latencia_p95_s} s | maxima: {resumen.latencia_maxima_s} s")
    print(f"La API aceptó cada petición en {resumen.aceptacion_maxima_s} s como máximo (no espera al modelo).")
    print(f"Tokens por ejecucion (promedio): {resumen.tokens_por_ejecucion_promedio}")
    if resumen.costo_por_ejecucion_usd is not None:
        print(f"Costo por ejecución: USD {resumen.costo_por_ejecucion_usd}")
    else:
        print("Costo por ejecución: no calculado acá (pasá --precio-entrada y --precio-salida); mirá el dashboard de Phoenix.")


def parsear(argumentos: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000", help="dirección de la API")
    parser.add_argument("--n", type=int, default=5, help="cantidad de peticiones concurrentes (por defecto 5)")
    parser.add_argument("--perfil", choices=sorted(TAREAS_POR_PERFIL), default="sin-aprobacion")
    parser.add_argument("--tiempo-maximo", type=float, default=300.0, help="segundos maximos que se espera a cada job")
    parser.add_argument("--precio-entrada", type=float, default=None, help="USD por millon de tokens de entrada del modelo")
    parser.add_argument("--precio-salida", type=float, default=None, help="USD por millon de tokens de salida del modelo")
    parser.add_argument("--salida", type=Path, default=None, help="archivo JSON del resumen (por defecto docs/evidencia/carga-<fecha>.json)")
    return parser.parse_args(argumentos)


async def main(argumentos: list[str] | None = None) -> int:
    args = parsear(argumentos)
    if args.n < 1:
        print("--n tiene que ser al menos 1", file=sys.stderr)
        return 2
    base = TAREAS_POR_PERFIL[args.perfil]
    tareas = [base[i % len(base)] for i in range(args.n)]
    inicio = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"Disparando {args.n} peticiones concurrentes a {args.url} (perfil: {args.perfil})...")
    try:
        async with httpx.AsyncClient(base_url=args.url, timeout=5.0) as sonda:
            await sonda.get("/health")
    except httpx.HTTPError:
        print(f"No hay una API respondiendo en {args.url}. Levantala primero (ver README).", file=sys.stderr)
        return 1
    resultados, segundos = await correr_prueba(
        args.url, tareas, aprobar=args.perfil == "con-aprobacion", tiempo_maximo=args.tiempo_maximo
    )
    resumen = resumir(
        resultados, url=args.url, perfil=args.perfil, inicio=inicio, segundos_de_pared=segundos,
        precio_entrada=args.precio_entrada, precio_salida=args.precio_salida,
    )
    imprimir(resumen)
    salida = args.salida or CARPETA_EVIDENCIA / f"carga-{datetime.now():%Y%m%d-%H%M%S}.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(asdict(resumen), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResumen guardado en {salida.relative_to(RAIZ) if salida.is_relative_to(RAIZ) else salida}")
    return 0 if resumen.fallidas == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
