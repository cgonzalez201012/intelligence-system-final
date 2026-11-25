"""Estado de los jobs y cola de tareas sobre Redis.

Dos estructuras, ambas en Redis para que la API y los workers (que son procesos distintos)
compartan todo sin depender de la memoria de ninguno:

    job:{id}          hash con el registro del job (estado, tarea, resultado, error, ...)
    cola:tareas       lista: la API hace LPUSH y los workers BRPOP (FIFO)
    job:{id}:reclamo  clave que se crea con SET NX al aprobar: asegura que una aprobacion se use UNA vez

Regla de orden: SIEMPRE se registra el job (estado PENDING) ANTES de encolarlo. Si fuera al reves,
un worker rapido podria tomar la tarea antes de que exista el registro y el cliente veria un 404.
"""

import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from pydantic import BaseModel
from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.schemas import EstadoJob, MensajeDeCola, TareaEstado

COLA_DE_TAREAS = "cola:tareas"
MAX_RECUPERACIONES = 2  # veces que se reencola una reanudacion interrumpida antes de darla por fallida
_ID_VALIDO = re.compile(r"[0-9a-f]{32}")  # los job_id los genera uuid4().hex: cualquier otra cosa no es un job
_CAMPOS_JSON = ("resultado", "aprobacion_pendiente", "decision", "uso")


def ahora() -> str:
    """Instante actual en UTC, en ISO 8601 con milisegundos."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _clave(job_id: str) -> str:
    return f"job:{job_id}"


class AlmacenDeJobs:
    """Acceso asincrono al estado de los jobs y a la cola. El cliente de Redis debe usar decode_responses=True."""

    def __init__(self, redis: Redis, *, ttl_segundos: int, cola: str = COLA_DE_TAREAS) -> None:
        self._redis = redis
        self._ttl = ttl_segundos
        self._cola = cola

    # ---------- Registro del job ----------

    async def crear(self, tarea: str) -> str:
        """Registra un job nuevo en estado PENDING y devuelve su ID (todavia NO lo encola)."""
        job_id = uuid.uuid4().hex
        instante = ahora()
        clave = _clave(job_id)
        async with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.hset(
                clave,
                mapping={
                    "job_id": job_id,
                    "estado": "PENDING",
                    "tarea": tarea,
                    "creado_en": instante,
                    "actualizado_en": instante,
                },
            )
            pipeline.expire(clave, self._ttl)
            await pipeline.execute()
        return job_id

    async def obtener(self, job_id: str) -> TareaEstado | None:
        """El registro del job, o None si no existe (o ya expiro, o el ID no tiene el formato de un job_id)."""
        if not _ID_VALIDO.fullmatch(job_id):
            return None  # evita consultar claves ajenas (p. ej. `job:<id>:reclamo`, que no es un hash)
        datos = cast("dict[str, str]", await self._redis.hgetall(_clave(job_id)))
        if not datos:
            return None
        crudo: dict[str, Any] = dict(datos)
        for campo in _CAMPOS_JSON:
            if campo in crudo:
                crudo[campo] = json.loads(crudo[campo]) if crudo[campo] else None
        return TareaEstado.model_validate(crudo)

    async def actualizar(self, job_id: str, estado: EstadoJob, **campos: Any) -> None:
        """Cambia el estado del job y guarda los campos que se indiquen (los JSON se serializan solos).

        Los campos con valor None se borran del registro (por ejemplo, `aprobacion_pendiente` al reanudar).
        """
        instante = ahora()
        a_guardar: dict[Any, Any] = {"estado": estado, "actualizado_en": instante}
        a_borrar: list[str] = []
        if estado in ("DONE", "FAILED"):
            a_guardar["finalizado_en"] = instante
        for nombre, valor in campos.items():
            if valor is None:
                a_borrar.append(nombre)
            elif nombre in _CAMPOS_JSON:
                if isinstance(valor, BaseModel):
                    valor = valor.model_dump(mode="json")
                a_guardar[nombre] = json.dumps(valor, ensure_ascii=False)
            else:
                a_guardar[nombre] = str(valor)
        clave = _clave(job_id)
        async with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.hset(clave, mapping=a_guardar)
            if a_borrar:
                pipeline.hdel(clave, *a_borrar)
            pipeline.expire(clave, self._ttl)
            await pipeline.execute()

    async def rescatar_reanudaciones_huerfanas(
        self, *, mas_viejos_que_segundos: float, max_recuperaciones: int = MAX_RECUPERACIONES
    ) -> list[str]:
        """Devuelve a la cola los jobs que murieron a mitad de una REANUDACION, en vez de darlos por perdidos.

        Un job que ya recibio la decision humana tiene su checkpoint guardado en Redis (quedo pausado ahi):
        si el worker que lo reanudaba se cae, no hay que pedirle a la persona que apruebe de nuevo ni repetir el
        analisis; alcanza con reencolar la reanudacion para que otro worker la retome. Es seguro porque el
        escalamiento no se duplica (ver `NotificadorRedis`). Se hace en UNA transaccion (estado + cola) para no
        dejar un job en PENDING sin mensaje. Pasadas `max_recuperaciones` el job se deja para `marcar_huerfanos`.
        """
        limite = datetime.now(UTC) - timedelta(seconds=mas_viejos_que_segundos)
        rescatados: list[str] = []
        async for clave in self._redis.scan_iter(match="job:*", count=200):
            job_id = str(clave)[len("job:"):]
            if not _ID_VALIDO.fullmatch(job_id):
                continue
            async with self._redis.pipeline(transaction=True) as pipeline:
                try:
                    await pipeline.watch(clave)
                    estado, actualizado, decision, recuperaciones = await pipeline.hmget(
                        clave, ["estado", "actualizado_en", "decision", "recuperaciones"]
                    )
                    if estado != "RUNNING" or not decision or not actualizado:
                        continue
                    if datetime.fromisoformat(str(actualizado)) > limite or int(recuperaciones or 0) >= max_recuperaciones:
                        continue
                    pipeline.multi()
                    pipeline.hset(clave, mapping={
                        "estado": "PENDING", "actualizado_en": ahora(), "recuperaciones": int(recuperaciones or 0) + 1,
                    })
                    pipeline.expire(clave, self._ttl)
                    pipeline.lpush(self._cola, MensajeDeCola(job_id=job_id, accion="reanudar").model_dump_json())
                    await pipeline.execute()
                    rescatados.append(job_id)
                except (WatchError, ValueError):
                    continue
        return rescatados

    async def marcar_huerfanos(self, *, mas_viejos_que_segundos: float, motivo: str) -> list[str]:
        """Pasa a FAILED los jobs que llevan demasiado tiempo en RUNNING (su worker se cayo y nadie los va a terminar).

        Sin esto, un worker que muere a mitad de un job (kill, falta de memoria, reinicio del contenedor) deja
        el job en RUNNING para siempre y el cliente hace polling sin fin. Cada job se revisa y se cambia dentro
        de una transaccion con WATCH: si el job cambia de estado justo en ese momento, no se toca.
        """
        limite = datetime.now(UTC) - timedelta(seconds=mas_viejos_que_segundos)
        marcados: list[str] = []
        async for clave in self._redis.scan_iter(match="job:*", count=200):
            job_id = str(clave)[len("job:"):]
            if not _ID_VALIDO.fullmatch(job_id):
                continue  # claves auxiliares (job:<id>:reclamo)
            async with self._redis.pipeline(transaction=True) as pipeline:
                try:
                    await pipeline.watch(clave)
                    estado, actualizado = await pipeline.hmget(clave, ["estado", "actualizado_en"])
                    if estado != "RUNNING" or not actualizado or datetime.fromisoformat(str(actualizado)) > limite:
                        continue
                    instante = ahora()
                    pipeline.multi()
                    pipeline.hset(clave, mapping={
                        "estado": "FAILED", "error": motivo, "actualizado_en": instante, "finalizado_en": instante,
                    })
                    pipeline.expire(clave, self._ttl)
                    await pipeline.execute()
                    marcados.append(job_id)
                except (WatchError, ValueError):
                    continue
        return marcados

    # ---------- Cola ----------

    async def encolar(self, mensaje: MensajeDeCola) -> None:
        await self._redis.lpush(self._cola, mensaje.model_dump_json())

    async def tomar_de_la_cola(self, espera_segundos: float = 1.0) -> MensajeDeCola | None:
        """Saca el proximo mensaje (espera hasta `espera_segundos`); None si no llego nada."""
        elemento = await self._redis.brpop([self._cola], timeout=espera_segundos)
        if elemento is None:
            return None
        _, contenido = elemento
        return MensajeDeCola.model_validate_json(contenido)

    async def largo_de_la_cola(self) -> int:
        return int(await self._redis.llen(self._cola))

    # ---------- Aprobaciones ----------

    async def reclamar_aprobacion(self, job_id: str) -> bool:
        """True solo para el PRIMER que llama: dos aprobaciones simultaneas no pueden reanudar dos veces."""
        creado = await self._redis.set(f"{_clave(job_id)}:reclamo", "1", nx=True, ex=self._ttl)
        return bool(creado)

    async def liberar_aprobacion(self, job_id: str) -> None:
        """Libera el reclamo (cuando se pudo tomar la aprobacion pero fallo el reencolado)."""
        await self._redis.delete(f"{_clave(job_id)}:reclamo")

    # ---------- Salud ----------

    async def responde(self) -> bool:
        return bool(await self._redis.ping())
