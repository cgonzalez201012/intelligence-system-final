"""El efecto secundario real (anotar el escalamiento en Redis) y su idempotencia."""

import json

from redis.asyncio import Redis

from app.notificador import BANDEJA_DE_SALIDA, NotificadorRedis
from orquestador.state import AccionPropuesta


def _accion(incidente: str = "INC-1001") -> AccionPropuesta:
    return AccionPropuesta(
        incidente_id=incidente, severidad="critica", destinatario="Gerente Comercial", exceso_horas=276, motivo="vencido"
    )


async def test_notificar_anota_el_aviso_en_la_bandeja_de_salida(redis: Redis) -> None:
    notificador = NotificadorRedis(redis, ttl_segundos=60)
    resultado = await notificador.notificar("job-1", _accion())
    assert resultado.ejecutada is True and "Gerente Comercial" in resultado.detalle
    avisos = [json.loads(a) for a in await redis.lrange(BANDEJA_DE_SALIDA, 0, -1)]
    assert len(avisos) == 1
    assert avisos[0]["job_id"] == "job-1" and avisos[0]["incidente_id"] == "INC-1001"
    assert avisos[0]["destinatario"] == "Gerente Comercial"


async def test_notificar_dos_veces_lo_mismo_no_duplica_el_aviso(redis: Redis) -> None:
    """Si el worker se cae despues de enviar y antes del checkpoint, al reanudar el nodo vuelve a correr."""
    notificador = NotificadorRedis(redis, ttl_segundos=60)
    await notificador.notificar("job-1", _accion())
    segundo = await notificador.notificar("job-1", _accion())
    assert await redis.llen(BANDEJA_DE_SALIDA) == 1  # type: ignore[misc]
    assert segundo.ejecutada is True and "Ya estaba escalado" in segundo.detalle


async def test_el_mismo_incidente_en_otro_job_si_se_notifica(redis: Redis) -> None:
    notificador = NotificadorRedis(redis, ttl_segundos=60)
    await notificador.notificar("job-1", _accion())
    await notificador.notificar("job-2", _accion())
    assert await redis.llen(BANDEJA_DE_SALIDA) == 2  # type: ignore[misc]
