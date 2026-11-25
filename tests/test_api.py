"""La API: endpoints asincronos que encolan y responden al instante, sin ejecutar nada de IA."""

import asyncio
import time

import httpx
import pytest
from redis.exceptions import RedisError

from app.jobs import AlmacenDeJobs
from app.schemas import MensajeDeCola

TAREA_VALIDA = "Priorizá los incidentes críticos abiertos del maestro de artículos."
APROBACION = {"aprobada": True, "revisor": "Cris", "comentario": "ok"}


# ---------- POST /tasks ----------


async def test_post_tasks_encola_y_devuelve_el_job_id_al_instante(cliente: httpx.AsyncClient, almacen: AlmacenDeJobs) -> None:
    inicio = time.perf_counter()
    respuesta = await cliente.post("/tasks", json={"tarea": TAREA_VALIDA})
    assert time.perf_counter() - inicio < 1  # no corre el agente: solo registra y encola
    assert respuesta.status_code == 202
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "PENDING" and len(cuerpo["job_id"]) == 32
    assert respuesta.headers["location"] == f"/tasks/{cuerpo['job_id']}"
    assert await almacen.largo_de_la_cola() == 1
    mensaje = await almacen.tomar_de_la_cola(0.1)
    assert mensaje == MensajeDeCola(job_id=cuerpo["job_id"], accion="ejecutar")


async def test_el_job_ya_existe_en_redis_cuando_se_encola(almacen: AlmacenDeJobs, cliente: httpx.AsyncClient) -> None:
    """Error comun de la consigna: si el worker fuera mas rapido que el registro, el cliente veria un 404."""
    visto: list[str | None] = []
    encolar_original = almacen.encolar

    async def espiar(mensaje: MensajeDeCola) -> None:
        job = await almacen.obtener(mensaje.job_id)  # lo que veria un worker en el instante de recibir la tarea
        visto.append(job.estado if job else None)
        await encolar_original(mensaje)

    almacen.encolar = espiar  # type: ignore[method-assign]
    await cliente.post("/tasks", json={"tarea": TAREA_VALIDA})
    assert visto == ["PENDING"]


@pytest.mark.parametrize("cuerpo", [{}, {"tarea": ""}, {"tarea": "corta"}, {"tarea": "x" * 2001}, {"tarea": 123}, {"otro": "campo"}])
async def test_post_tasks_rechaza_tareas_invalidas_sin_encolar(cliente: httpx.AsyncClient, almacen: AlmacenDeJobs, cuerpo: dict[str, object]) -> None:
    respuesta = await cliente.post("/tasks", json=cuerpo)
    assert respuesta.status_code == 422
    assert await almacen.largo_de_la_cola() == 0


async def test_post_tasks_normaliza_los_espacios_de_la_tarea(cliente: httpx.AsyncClient) -> None:
    creada = (await cliente.post("/tasks", json={"tarea": "  Priorizá   los incidentes\n\nabiertos  "})).json()
    assert (await cliente.get(f"/tasks/{creada['job_id']}")).json()["tarea"] == "Priorizá los incidentes abiertos"


async def test_si_redis_falla_al_encolar_el_job_queda_en_failed_y_la_api_responde_503(
    cliente: httpx.AsyncClient, almacen: AlmacenDeJobs
) -> None:
    async def encolar_roto(mensaje: MensajeDeCola) -> None:
        raise RedisError("cola caida")

    almacen.encolar = encolar_roto  # type: ignore[method-assign]
    respuesta = await cliente.post("/tasks", json={"tarea": TAREA_VALIDA})
    assert respuesta.status_code == 503
    # el job no quedo PENDING para siempre
    claves = [k async for k in almacen._redis.scan_iter("job:*")]  # noqa: SLF001
    assert len(claves) == 1
    job = await almacen.obtener(claves[0].removeprefix("job:"))
    assert job is not None and job.estado == "FAILED" and "encolar" in (job.error or "")


async def test_la_api_atiende_cinco_pedidos_concurrentes_sin_bloquearse(cliente: httpx.AsyncClient, almacen: AlmacenDeJobs) -> None:
    inicio = time.perf_counter()
    respuestas = await asyncio.gather(*[cliente.post("/tasks", json={"tarea": f"{TAREA_VALIDA} (consulta {i})"}) for i in range(5)])
    assert time.perf_counter() - inicio < 2
    assert [r.status_code for r in respuestas] == [202] * 5
    assert len({r.json()["job_id"] for r in respuestas}) == 5
    assert await almacen.largo_de_la_cola() == 5


# ---------- GET /tasks/{id} ----------


async def test_get_tasks_devuelve_el_registro_del_job(cliente: httpx.AsyncClient) -> None:
    job_id = (await cliente.post("/tasks", json={"tarea": TAREA_VALIDA})).json()["job_id"]
    respuesta = await cliente.get(f"/tasks/{job_id}")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "PENDING" and cuerpo["tarea"] == TAREA_VALIDA
    assert cuerpo["resultado"] is None and cuerpo["error"] is None


async def test_get_tasks_de_un_job_inexistente_es_404(cliente: httpx.AsyncClient) -> None:
    assert (await cliente.get("/tasks/no-existe")).status_code == 404


async def test_el_estado_sobrevive_a_un_almacen_nuevo(cliente: httpx.AsyncClient, almacen: AlmacenDeJobs) -> None:
    """El estado vive en Redis, no en la memoria de la API: otra instancia (otro proceso) lo ve igual."""
    job_id = (await cliente.post("/tasks", json={"tarea": TAREA_VALIDA})).json()["job_id"]
    otro = AlmacenDeJobs(almacen._redis, ttl_segundos=60)  # noqa: SLF001
    job = await otro.obtener(job_id)
    assert job is not None and job.estado == "PENDING"


async def test_health_informa_que_redis_responde(cliente: httpx.AsyncClient) -> None:
    await cliente.post("/tasks", json={"tarea": TAREA_VALIDA})
    respuesta = await cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "redis": "ok", "tareas_en_cola": 1}


# ---------- POST /tasks/{id}/approve ----------


async def _job_esperando_aprobacion(almacen: AlmacenDeJobs) -> str:
    job_id = await almacen.crear(TAREA_VALIDA)
    await almacen.actualizar(job_id, "WAITING_APPROVAL", aprobacion_pendiente={"tipo": "aprobacion_requerida", "motivo": "m", "informe": "i", "acciones": [{"incidente_id": "INC-1001", "severidad": "critica", "destinatario": "Gerente Comercial", "exceso_horas": 5, "motivo": "x"}]})
    return job_id


async def test_approve_registra_la_decision_y_encola_la_reanudacion(cliente: httpx.AsyncClient, almacen: AlmacenDeJobs) -> None:
    job_id = await _job_esperando_aprobacion(almacen)
    respuesta = await cliente.post(f"/tasks/{job_id}/approve", json=APROBACION)
    assert respuesta.status_code == 202
    assert respuesta.json()["estado"] == "PENDING"  # vuelve a la cola hasta que un worker lo retome
    job = await almacen.obtener(job_id)
    assert job is not None and job.decision is not None and job.decision.revisor == "Cris"
    assert job.aprobacion_pendiente is None
    assert await almacen.tomar_de_la_cola(0.1) == MensajeDeCola(job_id=job_id, accion="reanudar")


async def test_approve_de_un_job_inexistente_es_404(cliente: httpx.AsyncClient) -> None:
    assert (await cliente.post("/tasks/no-existe/approve", json=APROBACION)).status_code == 404


@pytest.mark.parametrize("estado", ["PENDING", "RUNNING", "DONE", "FAILED"])
async def test_approve_solo_vale_para_un_job_en_waiting_approval(cliente: httpx.AsyncClient, almacen: AlmacenDeJobs, estado: str) -> None:
    job_id = await almacen.crear(TAREA_VALIDA)
    await almacen.actualizar(job_id, estado)  # type: ignore[arg-type]
    respuesta = await cliente.post(f"/tasks/{job_id}/approve", json=APROBACION)
    assert respuesta.status_code == 409 and estado in respuesta.json()["detail"]
    assert await almacen.largo_de_la_cola() == 0


async def test_una_segunda_aprobacion_del_mismo_job_es_409(cliente: httpx.AsyncClient, almacen: AlmacenDeJobs) -> None:
    job_id = await _job_esperando_aprobacion(almacen)
    assert (await cliente.post(f"/tasks/{job_id}/approve", json=APROBACION)).status_code == 202
    assert (await cliente.post(f"/tasks/{job_id}/approve", json=APROBACION)).status_code == 409
    assert await almacen.largo_de_la_cola() == 1  # una sola reanudacion


async def test_dos_aprobaciones_simultaneas_reanudan_una_sola_vez(cliente: httpx.AsyncClient, almacen: AlmacenDeJobs) -> None:
    job_id = await _job_esperando_aprobacion(almacen)
    respuestas = await asyncio.gather(*[cliente.post(f"/tasks/{job_id}/approve", json={**APROBACION, "revisor": f"persona{i}"}) for i in range(4)])
    assert sorted(r.status_code for r in respuestas) == [202, 409, 409, 409]
    assert await almacen.largo_de_la_cola() == 1


@pytest.mark.parametrize("cuerpo", [{}, {"aprobada": True}, {"revisor": "Cris"}, {"aprobada": "tal vez", "revisor": "Cris"}, {"aprobada": True, "revisor": ""}])
async def test_approve_rechaza_decisiones_mal_formadas(cliente: httpx.AsyncClient, almacen: AlmacenDeJobs, cuerpo: dict[str, object]) -> None:
    job_id = await _job_esperando_aprobacion(almacen)
    assert (await cliente.post(f"/tasks/{job_id}/approve", json=cuerpo)).status_code == 422
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "WAITING_APPROVAL"  # sigue esperando


async def test_si_redis_falla_al_aprobar_el_job_vuelve_a_esperar_y_se_puede_reintentar(
    cliente: httpx.AsyncClient, almacen: AlmacenDeJobs
) -> None:
    job_id = await _job_esperando_aprobacion(almacen)
    encolar_original = almacen.encolar

    async def encolar_roto(mensaje: MensajeDeCola) -> None:
        raise RedisError("cola caida")

    almacen.encolar = encolar_roto  # type: ignore[method-assign]
    assert (await cliente.post(f"/tasks/{job_id}/approve", json=APROBACION)).status_code == 503
    job = await almacen.obtener(job_id)
    assert job is not None and job.estado == "WAITING_APPROVAL" and job.aprobacion_pendiente is not None
    assert job.decision is None
    almacen.encolar = encolar_original  # type: ignore[method-assign]
    assert (await cliente.post(f"/tasks/{job_id}/approve", json=APROBACION)).status_code == 202  # el reclamo se libero
