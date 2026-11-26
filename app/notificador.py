"""Efecto secundario de las acciones criticas: escalar un incidente al responsable.

En este proyecto el envío es SIMULADO (la empresa y los datos son ficticios): "enviar" es
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

    async def _ya_esta_en_la_bandeja(self, job_id: str, incidente_id: str) -> bool:
        for crudo in await self._redis.lrange(self._bandeja, 0, -1):
            try:
                aviso = json.loads(crudo)
            except json.JSONDecodeError:
                continue
            if aviso.get("job_id") == job_id and aviso.get("incidente_id") == incidente_id:
                return True
        return False

    async def notificar(self, job_id: str, accion: AccionPropuesta) -> ResultadoAccion:
        # Si el worker se cae y al reanudar este nodo vuelve a correr, el aviso no puede salir dos veces NI perderse.
        # SET NX y LPUSH no son atomicos entre si, asi que la marca tiene dos estados: "pendiente" (se reservo el
        # envio) y "enviado" (el aviso ya esta en la bandeja). Si se cayo entre los dos, al reintentar se mira la
        # bandeja: si el aviso esta, no se repite; si no, se envia ahora.
        clave = f"escalacion:{job_id}:{accion.incidente_id}"
        es_nuevo = await self._redis.set(clave, "pendiente", nx=True, ex=self._ttl)
        if not es_nuevo:
            enviado = await self._redis.get(clave) == "enviado"
            if enviado or await self._ya_esta_en_la_bandeja(job_id, accion.incidente_id):
                if not enviado:
                    await self._redis.set(clave, "enviado", ex=self._ttl)
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
        await self._redis.set(clave, "enviado", ex=self._ttl)
        return ResultadoAccion(
            incidente_id=accion.incidente_id,
            destinatario=accion.destinatario,
            ejecutada=True,
            detalle=f"Escalamiento registrado en la bandeja de salida para {accion.destinatario}.",
        )
