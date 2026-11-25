"""Un worker se cae con el job pausado (o reanudándolo): otro lo retoma desde el checkpoint de Redis.

"Caerse" se simula descartando por completo el worker (su grafo, sus modelos, su memoria) y armando OTRO
desde cero que solo comparte lo que en producción vive en Redis: la cola, el registro de los jobs y los
checkpoints. Los checkpoints pasan por el serializador real del estado, como en Redis: lo que el segundo worker
recibe se reconstruye desde bytes, no es el mismo objeto en memoria.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from redis.asyncio import Redis

from app.graph import serializador_del_estado
from app.jobs import MAX_RECUPERACIONES, AlmacenDeJobs
from app.schemas import MensajeDeCola
from app.settings import Ajustes
from app.worker import MOTIVO_HUERFANO, Servicios, barrer_huerfanos, procesar_mensaje
from orquestador.state import Hallazgos
from tests.conftest import NotificadorEspia, enviar_y_procesar, reanudar_y_procesar
from tests.fakes import SupervisorFalso, decidir, especialistas_falsos, guion_analista, guion_investigador, informe

APROBACION = {"aprobada": True, "revisor": "Cris", "comentario": "ok"}
HACE_UNA_HORA = (datetime.now(UTC) - timedelta(hours=1)).isoformat(timespec="milliseconds")


@pytest.fixture
def checkpoints() -> InMemorySaver:
    """El 'Redis' de los checkpoints: lo comparten los workers y serializa como lo hace el real."""
    return InMemorySaver(serde=serializador_del_estado())


def _worker_original(armar_grafo_produccion: Any, almacen: AlmacenDeJobs, ajustes: Ajustes, config: Any, checkpoints: Any) -> Servicios:
    supervisor = SupervisorFalso(
        guion=[decidir("investigador", "Recolectá."), decidir("analista", "Calculá."), decidir("redactor", "Escribí."), decidir("FINISH")]
    )
    modelos = especialistas_falsos(guion_investigador(), guion_analista(), [informe("INC-1001", "INC-1002")])
    grafo = armar_grafo_produccion(supervisor, modelos, checkpointer=checkpoints)
    return Servicios(almacen=almacen, grafo=grafo, ajustes=ajustes, config=config)


def _worker_de_reemplazo(armar_grafo_produccion: Any, almacen: AlmacenDeJobs, ajustes: Ajustes, config: Any, checkpoints: Any):
    """Un worker nuevo y SIN guion: si llamara a algún modelo para rehacer el trabajo, el test falla."""
    supervisor = SupervisorFalso(guion=[decidir("FINISH")])
    modelos = especialistas_falsos()
    grafo = armar_grafo_produccion(supervisor, modelos, checkpointer=checkpoints)
    return Servicios(almacen=almacen, grafo=grafo, ajustes=ajustes, config=config), supervisor, modelos


async def test_si_el_worker_se_cae_con_el_job_pausado_otro_lo_retoma_sin_rehacer_el_trabajo(
    armar_grafo_produccion, almacen, ajustes, config, checkpoints, cliente: httpx.AsyncClient, espia: NotificadorEspia
) -> None:
    original = _worker_original(armar_grafo_produccion, almacen, ajustes, config, checkpoints)
    job_id = await enviar_y_procesar(original)
    assert (await almacen.obtener(job_id)).estado == "WAITING_APPROVAL"  # type: ignore[union-attr]
    assert espia.llamadas == []  # pausado: nada se ejecutó todavía

    del original  # el worker se cae: se pierde todo lo que tenía en memoria

    # La persona aprueba mientras NO hay worker: la decisión y la reanudación quedan guardadas en Redis.
    assert (await cliente.post(f"/tasks/{job_id}/approve", json=APROBACION)).status_code == 202
    assert (await almacen.obtener(job_id)).estado == "PENDING"  # type: ignore[union-attr]

    reemplazo, supervisor, modelos = _worker_de_reemplazo(armar_grafo_produccion, almacen, ajustes, config, checkpoints)
    await reanudar_y_procesar(reemplazo)

    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "DONE" and job.resultado is not None
    assert job.resultado.estado_final == "completado" and "INC-1001" in job.resultado.respuesta
    assert sorted(a.incidente_id for a in job.resultado.acciones.ejecutadas) == ["INC-1001", "INC-1002"]
    assert sorted(i for _, i in espia.llamadas) == ["INC-1001", "INC-1002"]  # cada escalamiento, una sola vez
    assert supervisor.recibidos == [] and modelos.recibidos == []  # no se volvió a llamar a ningún modelo


async def test_el_estado_retomado_conserva_sus_tipos_porque_se_reconstruye_desde_el_checkpoint(
    armar_grafo_produccion, almacen, ajustes, config, checkpoints
) -> None:
    original = _worker_original(armar_grafo_produccion, almacen, ajustes, config, checkpoints)
    job_id = await enviar_y_procesar(original)
    reemplazo, _, _ = _worker_de_reemplazo(armar_grafo_produccion, almacen, ajustes, config, checkpoints)
    foto = await reemplazo.grafo.aget_state({"configurable": {"thread_id": job_id}})
    assert foto.next == ("aprobacion_humana",) and foto.interrupts
    assert isinstance(foto.values["hallazgos"], Hallazgos)  # un modelo Pydantic, no un dict suelto
    assert [a.incidente_id for a in foto.values["acciones_propuestas"]] == ["INC-1001", "INC-1002"]


async def test_si_el_worker_se_cae_mientras_reanuda_el_barrido_devuelve_el_job_a_la_cola_y_otro_lo_termina(
    armar_grafo_produccion, almacen: AlmacenDeJobs, redis: Redis, ajustes, config, checkpoints, cliente: httpx.AsyncClient,
    espia: NotificadorEspia,
) -> None:
    original = _worker_original(armar_grafo_produccion, almacen, ajustes, config, checkpoints)
    job_id = await enviar_y_procesar(original)
    await cliente.post(f"/tasks/{job_id}/approve", json=APROBACION)

    # El worker saca el mensaje de reanudación de la cola y muere antes de terminar: el job queda RUNNING.
    mensaje = await almacen.tomar_de_la_cola(0.1)
    assert mensaje is not None and mensaje.accion == "reanudar"
    await almacen.actualizar(job_id, "RUNNING")
    await redis.hset(f"job:{job_id}", "actualizado_en", HACE_UNA_HORA)
    del original

    rescatados = await almacen.rescatar_reanudaciones_huerfanas(mas_viejos_que_segundos=360)
    assert rescatados == [job_id]
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "PENDING" and job.recuperaciones == 1
    assert job.decision is not None and job.decision.revisor == "Cris"  # la decisión humana no se pierde

    reemplazo, _, _ = _worker_de_reemplazo(armar_grafo_produccion, almacen, ajustes, config, checkpoints)
    await reanudar_y_procesar(reemplazo)
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "DONE" and job.resultado is not None
    assert sorted(i for _, i in espia.llamadas) == ["INC-1001", "INC-1002"]


async def test_la_reanudacion_rescatada_no_duplica_los_avisos_ya_enviados(
    redis: Redis, almacen: AlmacenDeJobs, ajustes: Ajustes, config, armar_grafo_produccion, checkpoints, cliente: httpx.AsyncClient
) -> None:
    """Muere DESPUÉS de enviar los avisos y ANTES de guardar el checkpoint: al reanudar, la bandeja no se repite."""
    import json

    from app.graph import construir_grafo_produccion
    from app.notificador import BANDEJA_DE_SALIDA, NotificadorRedis

    def worker() -> Servicios:
        supervisor = SupervisorFalso(guion=[decidir("investigador", "R"), decidir("analista", "C"), decidir("redactor", "E"), decidir("FINISH")])
        modelos = especialistas_falsos(guion_investigador(), guion_analista(), [informe("INC-1001", "INC-1002")])
        grafo = construir_grafo_produccion(
            supervisor, modelos, config, checkpointer=checkpoints, notificador=NotificadorRedis(redis, ttl_segundos=60)
        )
        return Servicios(almacen=almacen, grafo=grafo, ajustes=ajustes, config=config)

    job_id = await enviar_y_procesar(worker())
    await cliente.post(f"/tasks/{job_id}/approve", json=APROBACION)
    # Simula que el worker ya mandó los avisos antes de caerse (dejó las marcas de "ya escalado" y la bandeja escrita).
    notificador = NotificadorRedis(redis, ttl_segundos=60)
    foto = await worker().grafo.aget_state({"configurable": {"thread_id": job_id}})
    for accion in foto.values["acciones_propuestas"]:
        await notificador.notificar(job_id, accion)
    assert await redis.llen(BANDEJA_DE_SALIDA) == 2  # type: ignore[misc]

    await reanudar_y_procesar(worker())  # el reemplazo reanuda y vuelve a pasar por la ejecución
    avisos = [json.loads(a) for a in await redis.lrange(BANDEJA_DE_SALIDA, 0, -1)]  # type: ignore[misc]
    assert sorted(a["incidente_id"] for a in avisos) == ["INC-1001", "INC-1002"]  # siguen siendo 2, no 4
    assert (await almacen.obtener(job_id)).estado == "DONE"  # type: ignore[union-attr]


async def test_una_reanudacion_que_se_cae_una_y_otra_vez_termina_en_failed_y_no_en_un_bucle(
    almacen: AlmacenDeJobs, redis: Redis
) -> None:
    job_id = await almacen.crear("Tarea que se cae siempre al reanudar")
    await redis.hset(f"job:{job_id}", mapping={
        "estado": "RUNNING", "actualizado_en": HACE_UNA_HORA, "decision": '{"aprobada": true, "revisor": "Cris", "comentario": ""}',
    })
    for _ in range(MAX_RECUPERACIONES):
        assert await almacen.rescatar_reanudaciones_huerfanas(mas_viejos_que_segundos=360) == [job_id]
        await almacen.tomar_de_la_cola(0.1)  # un worker lo toma y vuelve a morir
        await almacen.actualizar(job_id, "RUNNING")
        await redis.hset(f"job:{job_id}", "actualizado_en", HACE_UNA_HORA)
    assert await almacen.rescatar_reanudaciones_huerfanas(mas_viejos_que_segundos=360) == []  # ya no se rescata
    assert await almacen.marcar_huerfanos(mas_viejos_que_segundos=360, motivo=MOTIVO_HUERFANO) == [job_id]
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "FAILED" and job.recuperaciones == MAX_RECUPERACIONES


async def test_un_job_que_no_era_una_reanudacion_no_se_rescata_se_marca_failed(almacen: AlmacenDeJobs, redis: Redis) -> None:
    job_id = await almacen.crear("Tarea nueva que se cayó corriendo")
    await redis.hset(f"job:{job_id}", mapping={"estado": "RUNNING", "actualizado_en": HACE_UNA_HORA})  # sin decisión
    assert await almacen.rescatar_reanudaciones_huerfanas(mas_viejos_que_segundos=360) == []
    assert await almacen.marcar_huerfanos(mas_viejos_que_segundos=360, motivo=MOTIVO_HUERFANO) == [job_id]


async def test_los_jobs_vivos_o_en_pausa_no_se_tocan_al_rescatar(almacen: AlmacenDeJobs, redis: Redis) -> None:
    ahora = datetime.now(UTC).isoformat(timespec="milliseconds")
    decision = '{"aprobada": true, "revisor": "Cris", "comentario": ""}'
    vivo = await almacen.crear("Reanudación en curso y reciente")
    await redis.hset(f"job:{vivo}", mapping={"estado": "RUNNING", "actualizado_en": ahora, "decision": decision})
    pausado = await almacen.crear("Pausado esperando aprobación")
    await redis.hset(f"job:{pausado}", mapping={"estado": "WAITING_APPROVAL", "actualizado_en": HACE_UNA_HORA})
    assert await almacen.rescatar_reanudaciones_huerfanas(mas_viejos_que_segundos=360) == []
    assert (await almacen.obtener(pausado)).estado == "WAITING_APPROVAL"  # type: ignore[union-attr]
    assert await almacen.largo_de_la_cola() == 0


async def test_el_barrido_del_worker_rescata_y_el_mensaje_queda_listo_para_otro_worker(
    almacen: AlmacenDeJobs, redis: Redis, ajustes: Ajustes
) -> None:
    import asyncio

    job_id = await almacen.crear("Reanudación huérfana")
    await redis.hset(f"job:{job_id}", mapping={
        "estado": "RUNNING", "actualizado_en": HACE_UNA_HORA, "decision": '{"aprobada": true, "revisor": "Cris", "comentario": ""}',
    })
    servicios = Servicios(almacen=almacen, grafo=None, ajustes=ajustes, config=None)  # type: ignore[arg-type]
    detener = asyncio.Event()
    tarea = asyncio.create_task(barrer_huerfanos(servicios, detener, 0.05))
    await asyncio.sleep(0.2)
    detener.set()
    await tarea
    assert (await almacen.obtener(job_id)).estado == "PENDING"  # type: ignore[union-attr]
    assert await almacen.tomar_de_la_cola(0.1) == MensajeDeCola(job_id=job_id, accion="reanudar")


# --------------------------------------------------------------------------- el aviso no se duplica NI se pierde
async def _accion():
    from orquestador.state import AccionPropuesta

    return AccionPropuesta(incidente_id="INC-1001", severidad="critica", destinatario="Gerente", exceso_horas=5, motivo="m")


async def test_si_el_worker_cayo_tras_reservar_el_envio_y_antes_de_avisar_el_aviso_sale_al_reintentar(redis: Redis) -> None:
    from app.notificador import BANDEJA_DE_SALIDA, NotificadorRedis

    notificador = NotificadorRedis(redis, ttl_segundos=60)
    await redis.set("escalacion:job1:INC-1001", "pendiente", ex=60)   # reservó el envío y murió antes del LPUSH
    resultado = await notificador.notificar("job1", await _accion())
    assert resultado.ejecutada and "registrado" in resultado.detalle
    assert await redis.llen(BANDEJA_DE_SALIDA) == 1  # type: ignore[misc]  # el aviso NO se perdió
    assert (await notificador.notificar("job1", await _accion())).detalle.startswith("Ya estaba escalado")
    assert await redis.llen(BANDEJA_DE_SALIDA) == 1  # type: ignore[misc]  # y no se duplicó


async def test_si_el_worker_cayo_tras_avisar_y_antes_de_confirmar_no_se_repite_el_aviso(redis: Redis) -> None:
    import json

    from app.notificador import BANDEJA_DE_SALIDA, NotificadorRedis

    notificador = NotificadorRedis(redis, ttl_segundos=60)
    await redis.set("escalacion:job1:INC-1001", "pendiente", ex=60)
    await redis.lpush(BANDEJA_DE_SALIDA, json.dumps({"job_id": "job1", "incidente_id": "INC-1001"}))  # type: ignore[misc]
    resultado = await notificador.notificar("job1", await _accion())
    assert resultado.detalle.startswith("Ya estaba escalado")
    assert await redis.llen(BANDEJA_DE_SALIDA) == 1  # type: ignore[misc]
    assert await redis.get("escalacion:job1:INC-1001") == "enviado"  # quedó confirmado para las próximas veces
