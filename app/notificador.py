"""Efecto secundario de las acciones criticas: escalar un incidente al responsable.

En esta pre-entrega el envio es SIMULADO (la empresa y los datos son ficticios): "enviar" es
anotar el escalamiento en la bandeja de salida `escalaciones:enviadas` de Redis. Es una
herramienta con efecto externo de todas formas: lo que importa es que SOLO corre despues de
la aprobacion humana, y que correrla dos veces no duplica el aviso.
"""

import json
from typing import Protocol

from redis.asyncio import Redis

from app.jobs import ahora
from orquestador.state import AccionPropuesta, ResultadoAccion

BANDEJA_DE_SALIDA = "escalaciones:enviadas"


class Notificador(Protocol):
    """Lo que necesita la etapa de ejecucion: poder escalar una accion aprobada."""

    async def notificar(self, job_id: str, accion: AccionPropuesta) -> ResultadoAccion: ...


class NotificadorRedis:
    """Escala incidentes anotandolos en la bandeja de salida de Redis (cliente con decode_responses=True)."""

    def __init__(self, redis: Redis, *, ttl_segundos: int, bandeja: str = BANDEJA_DE_SALIDA) -> None:
        self._redis = redis
        self._ttl = ttl_segundos
        self._bandeja = bandeja

    async def notificar(self, job_id: str, accion: AccionPropuesta) -> ResultadoAccion:
        # Si el worker se cae despues de enviar y antes de guardar el checkpoint, al reanudar el nodo
        # vuelve a correr: esta clave (SET NX) evita que el aviso salga dos veces.
        es_nuevo = await self._redis.set(f"escalacion:{job_id}:{accion.incidente_id}", "1", nx=True, ex=self._ttl)
        if not es_nuevo:
            return ResultadoAccion(
                incidente_id=accion.incidente_id,
                destinatario=accion.destinatario,
                ejecutada=True,
                detalle="Ya estaba escalado: no se envió un segundo aviso.",
            )
        aviso = {
            "job_id": job_id,
            "incidente_id": accion.incidente_id,
            "destinatario": accion.destinatario,
            "motivo": accion.motivo,
            "enviado_en": ahora(),
        }
        await self._redis.lpush(self._bandeja, json.dumps(aviso, ensure_ascii=False))
        return ResultadoAccion(
            incidente_id=accion.incidente_id,
            destinatario=accion.destinatario,
            ejecutada=True,
            detalle=f"Escalamiento registrado en la bandeja de salida para {accion.destinatario}.",
        )
