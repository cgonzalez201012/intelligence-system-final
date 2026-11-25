"""Human-in-the-loop: la pausa obligatoria antes de una accion con efecto secundario.

Se cuelga del CIERRE del orquestador del modulo 6, sin tocar la logica del supervisor:

    supervisor --FINISH--> proponer_acciones --(hay acciones)--> aprobacion_humana --(aprobada)--> ejecutar_acciones --> END
                                  |                                      |
                                  +--(no hay)--------> END               +--(rechazada)-------------------------> END

- `proponer_acciones` decide en CODIGO (no con el modelo) que tarea es critica: hay incidentes
  criticos abiertos con el SLA vencido, segun el analisis ya validado. Escalarlos avisa a una persona
  del negocio, es decir, tiene un efecto que no se deshace.
- `aprobacion_humana` llama a `interrupt(...)`: el grafo se detiene, LangGraph guarda el checkpoint
  en Redis y el worker queda libre. Se retoma cuando llega POST /tasks/{id}/approve, que reanuda el
  MISMO hilo con `Command(resume=decision)` (puede hacerlo otro worker: el estado esta en Redis).
- `ejecutar_acciones` es el unico nodo con efecto secundario, y revisa por su cuenta que haya una
  aprobacion antes de actuar (no confia solo en la arista que lo conecta).

`interrupt` hace que, al reanudar, el nodo se ejecute DE NUEVO desde su primera linea: por eso todo
lo que va antes de `interrupt` en `aprobacion_humana` es calculo puro, sin efectos.
"""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import interrupt
from pydantic import ValidationError
from redis.exceptions import RedisError

from app.dependencias import almacen_de
from app.jobs import AlmacenDeJobs
from app.notificador import Notificador
from app.schemas import AprobacionIn, ErrorApi, MensajeDeCola, TareaCreada
from orquestador.agents.research_agent import NodoAsincrono
from orquestador.state import AccionPropuesta, DecisionHumana, EstadoOrquestador, ResultadoAccion

NODO_PROPONER = "proponer_acciones"
NODO_APROBACION = "aprobacion_humana"
NODO_EJECUTAR = "ejecutar_acciones"

DESTINATARIO_POR_DEFECTO = "Comité de Gobierno de Datos"
MAX_CARACTERES_DEL_INFORME = 1500
MOTIVO_DE_LA_PAUSA = (
    "Hay incidentes críticos abiertos con el SLA vencido. Escalarlos avisa a su responsable (efecto externo): "
    "hace falta la aprobación de una persona."
)


# ---------- Que tarea es critica ----------


def calcular_acciones(estado: EstadoOrquestador) -> list[AccionPropuesta]:
    """Acciones a proponer: un escalamiento por cada incidente critico abierto con el SLA vencido.

    Sale solo del analisis que ya paso la validacion (la validacion recalcula los vencimientos),
    y solo si la corrida termino bien: un informe que no se pudo completar no escala nada.
    """
    analisis, hallazgos = estado["analisis"], estado["hallazgos"]
    if estado["estado_final"] != "completado" or analisis is None or hallazgos is None:
        return []
    fichas = {incidente.id: incidente for incidente in hallazgos.incidentes}
    vencimientos = {v.incidente_id: v for v in analisis.vencimientos}
    acciones: list[AccionPropuesta] = []
    for incidente_id in analisis.ranking:  # en orden de prioridad
        vencimiento, ficha = vencimientos.get(incidente_id), fichas.get(incidente_id)
        if vencimiento is None or ficha is None:
            continue
        if vencimiento.severidad != "critica" or not vencimiento.vencido or ficha.estado == "cerrado":
            continue
        acciones.append(
            AccionPropuesta(
                incidente_id=incidente_id,
                severidad=vencimiento.severidad,
                destinatario=ficha.responsable or DESTINATARIO_POR_DEFECTO,
                exceso_horas=vencimiento.exceso_horas,
                motivo=(
                    f"{incidente_id} ({ficha.titulo}) es crítico y lleva {vencimiento.exceso_horas} h hábiles "
                    f"de atraso sobre su SLA de {vencimiento.sla_horas_habiles} h."
                ),
            )
        )
    return acciones


# ---------- Nodos ----------


async def nodo_proponer_acciones(estado: EstadoOrquestador) -> dict[str, Any]:
    """Calcula las acciones criticas. No tiene efectos: solo lee el estado y propone."""
    return {"acciones_propuestas": calcular_acciones(estado)}


def decidir_tras_proponer(estado: EstadoOrquestador) -> Literal["aprobacion_humana", "__end__"]:
    """Hay acciones criticas: pausa para aprobacion. No hay: la tarea termina."""
    return NODO_APROBACION if estado["acciones_propuestas"] else END  # type: ignore[return-value]


def payload_de_la_pausa(estado: EstadoOrquestador) -> dict[str, Any]:
    """Lo que ve la persona que tiene que aprobar (JSON puro: viaja por el checkpoint y por la API)."""
    informe = (estado["respuesta_final"] or "")[:MAX_CARACTERES_DEL_INFORME]
    return {
        "tipo": "aprobacion_requerida",
        "motivo": MOTIVO_DE_LA_PAUSA,
        "acciones": [accion.model_dump(mode="json") for accion in estado["acciones_propuestas"]],
        "informe": informe,
    }


async def nodo_aprobacion_humana(estado: EstadoOrquestador) -> dict[str, Any]:
    """Pausa el grafo hasta recibir una decision humana desde afuera.

    Todo lo de aca arriba de `interrupt` se repite al reanudar: es calculo puro, sin efectos.
    """
    respuesta = interrupt(payload_de_la_pausa(estado))
    try:
        decision = DecisionHumana.model_validate(respuesta)
    except ValidationError:
        # Una senal ilegible nunca cuenta como un "si": ante la duda no se ejecuta nada.
        decision = DecisionHumana(
            aprobada=False, revisor="sistema", comentario="La respuesta de aprobación no tenía el formato pedido."
        )
    return {"aprobacion": decision}


def decidir_tras_aprobar(estado: EstadoOrquestador) -> Literal["ejecutar_acciones", "__end__"]:
    aprobacion = estado["aprobacion"]
    return NODO_EJECUTAR if aprobacion is not None and aprobacion.aprobada else END  # type: ignore[return-value]


def crear_nodo_ejecutar(notificador: Notificador) -> NodoAsincrono:
    """Nodo con el efecto secundario. Cada accion se ejecuta por separado y una falla no frena a las demas."""

    async def nodo_ejecutar_acciones(estado: EstadoOrquestador, config: RunnableConfig) -> dict[str, Any]:
        aprobacion = estado["aprobacion"]
        if aprobacion is None or not aprobacion.aprobada:
            return {"acciones_ejecutadas": []}  # segunda guarda: sin aprobacion no se hace nada
        job_id = str(config.get("configurable", {}).get("thread_id", ""))
        resultados: list[ResultadoAccion] = []
        for accion in estado["acciones_propuestas"]:
            try:
                resultados.append(await notificador.notificar(job_id, accion))
            except Exception as error:  # noqa: BLE001 - una accion que falla no debe tirar abajo el job entero
                resultados.append(
                    ResultadoAccion(
                        incidente_id=accion.incidente_id,
                        destinatario=accion.destinatario,
                        ejecutada=False,
                        detalle=f"No se pudo ejecutar ({type(error).__name__}).",
                    )
                )
        return {"acciones_ejecutadas": resultados}

    return nodo_ejecutar_acciones


# ---------- Endpoint de aprobacion ----------

router = APIRouter(prefix="/tasks", tags=["aprobación humana"])


@router.post(
    "/{job_id}/approve",
    status_code=202,
    response_model=TareaCreada,
    summary="Aprobar o rechazar las acciones críticas de un job pausado",
    responses={
        404: {"model": ErrorApi, "description": "No existe un job con ese ID (o ya expiró)."},
        409: {"model": ErrorApi, "description": "El job no está esperando una decisión, o ya recibió una."},
        503: {"model": ErrorApi, "description": "Redis no responde; la decisión no quedó registrada, se puede reintentar."},
    },
)
async def aprobar_tarea(
    job_id: str,
    decision: AprobacionIn,
    respuesta: Response,
    almacen: AlmacenDeJobs = Depends(almacen_de),
) -> TareaCreada:
    """Entrega la senal externa que reanuda el grafo. No lo corre: registra la decision y encola la reanudacion."""
    job = await almacen.obtener(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No existe un job con ese ID (o ya expiró).")
    if job.estado != "WAITING_APPROVAL":
        raise HTTPException(
            status_code=409,
            detail=f"El job está en estado {job.estado}: solo se puede decidir sobre uno en WAITING_APPROVAL.",
        )
    if not await almacen.reclamar_aprobacion(job_id):
        raise HTTPException(status_code=409, detail="Este job ya recibió una decisión.")
    try:
        # Mismo orden que en POST /tasks: primero el registro, despues la cola.
        await almacen.actualizar(
            job_id, "PENDING", decision=decision.model_dump(mode="json"), aprobacion_pendiente=None
        )
        await almacen.encolar(MensajeDeCola(job_id=job_id, accion="reanudar"))
    except RedisError as error:
        # Deshacer lo que se alcanzo a hacer para que la persona pueda volver a intentar.
        try:
            await almacen.actualizar(
                job_id, "WAITING_APPROVAL", aprobacion_pendiente=job.aprobacion_pendiente, decision=None
            )
            await almacen.liberar_aprobacion(job_id)
        except RedisError:
            pass  # Redis sigue caido: no hay mas que hacer desde aca; el 503 le avisa a la persona
        raise HTTPException(status_code=503, detail="No se pudo registrar la decisión: Redis no responde.") from error
    respuesta.headers["Location"] = f"/tasks/{job_id}"
    return TareaCreada(job_id=job_id, estado="PENDING", consultar_en=f"/tasks/{job_id}")
