"""Human-in-the-loop: la pausa antes de una accion con efecto secundario.

Lo central: sin una aprobacion, el efecto secundario NO ocurre (el notificador espia no se llama).
"""

from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.hitl import (
    NODO_APROBACION,
    NODO_EJECUTAR,
    NODO_PROPONER,
    calcular_acciones,
    crear_nodo_ejecutar,
    decidir_tras_aprobar,
    decidir_tras_proponer,
)
from orquestador.state import (
    CLAVES_POR_NODO,
    Analisis,
    DecisionHumana,
    Hallazgos,
    IncidenteHallazgo,
    VencimientoSLA,
    estado_inicial,
)
from tests.conftest import NotificadorEspia, crear_estado
from tests.fakes import (
    SupervisorFalso,
    decidir,
    especialistas_falsos,
    guion_analista,
    guion_investigador,
    informe,
)


def _supervisor() -> SupervisorFalso:
    return SupervisorFalso(
        guion=[decidir("investigador", "Recolectá."), decidir("analista", "Calculá."), decidir("redactor", "Escribí."), decidir("FINISH")]
    )


def _especialistas_criticos() -> Any:
    """Corrida con INC-1001 e INC-1002: criticos, abiertos y con el SLA vencido."""
    return especialistas_falsos(guion_investigador(), guion_analista(), [informe("INC-1001", "INC-1002")])


def _especialistas_sin_criticos() -> Any:
    """Corrida con INC-1003: severidad alta, no es critica."""
    return especialistas_falsos(
        guion_investigador(("INC-1003",), severidad="alta"), guion_analista(("INC-1003",)), [informe("INC-1003")]
    )


def _hilo(nombre: str = "job-1") -> dict[str, Any]:
    return {"configurable": {"thread_id": nombre}, "recursion_limit": 60}


APROBADA = {"aprobada": True, "revisor": "Cris", "comentario": "ok"}
RECHAZADA = {"aprobada": False, "revisor": "Cris", "comentario": "todavia no"}


# ---------- Estructura ----------


def test_el_grafo_de_produccion_cuelga_la_aprobacion_del_cierre(armar_grafo_produccion: Callable[..., CompiledStateGraph]) -> None:
    grafo = armar_grafo_produccion(_supervisor(), _especialistas_criticos())
    dibujo = grafo.get_graph()
    nodos = set(dibujo.nodes) - {"__start__", "__end__"}
    assert {"supervisor", "investigador", "analista", "redactor", "validacion", NODO_PROPONER, NODO_APROBACION, NODO_EJECUTAR} == nodos
    aristas = {(a.source, a.target) for a in dibujo.edges}
    assert ("supervisor", NODO_PROPONER) in aristas  # FINISH ya no va directo a END
    assert ("supervisor", "__end__") not in aristas
    assert (NODO_PROPONER, NODO_APROBACION) in aristas
    assert (NODO_APROBACION, NODO_EJECUTAR) in aristas
    assert (NODO_EJECUTAR, "__end__") in aristas
    # el unico camino al nodo con efecto secundario pasa por la aprobacion
    assert [a for a in aristas if a[1] == NODO_EJECUTAR] == [(NODO_APROBACION, NODO_EJECUTAR)]


def test_los_nodos_nuevos_escriben_solo_su_parcela() -> None:
    assert CLAVES_POR_NODO[NODO_PROPONER] == {"acciones_propuestas"}
    assert CLAVES_POR_NODO[NODO_APROBACION] == {"aprobacion"}
    assert CLAVES_POR_NODO[NODO_EJECUTAR] == {"acciones_ejecutadas"}


# ---------- Que tarea es critica (codigo, no modelo) ----------


def _estado_analizado(*, severidad: str = "critica", vencido: bool = True, estado: str = "abierto",
                      estado_final: str | None = "completado", responsable: str | None = "Gerente Comercial") -> Any:
    incidente = IncidenteHallazgo(
        id="INC-9", maestro="articulos", titulo="Articulos sin precio", severidad=severidad, estado=estado,  # type: ignore[arg-type]
        fecha_alta=date(2026, 9, 1), responsable=responsable, sla_horas_habiles=4,
    )
    vencimiento = VencimientoSLA(
        incidente_id="INC-9", severidad=severidad, fecha_alta=date(2026, 9, 1), fecha_corte=date(2026, 10, 31),  # type: ignore[arg-type]
        dias_habiles=40, horas_habiles_transcurridas=320, sla_horas_habiles=4, exceso_horas=316 if vencido else 0, vencido=vencido,
    )
    return crear_estado(
        hallazgos=Hallazgos(incidentes=[incidente]),
        analisis=Analisis(vencimientos=[vencimiento], ranking=["INC-9"]),
        estado_final=estado_final,
    )


def test_un_incidente_critico_vencido_dispara_una_accion_para_su_responsable() -> None:
    acciones = calcular_acciones(_estado_analizado())
    assert [(a.incidente_id, a.destinatario, a.exceso_horas) for a in acciones] == [("INC-9", "Gerente Comercial", 316)]
    assert "316" in acciones[0].motivo


@pytest.mark.parametrize(
    "cambios",
    [
        {"severidad": "alta"},  # no es critico
        {"vencido": False},  # critico pero dentro de plazo
        {"estado": "cerrado"},  # ya esta resuelto
        {"estado_final": "validacion_fallida"},  # la corrida no termino bien
        {"estado_final": None},
    ],
)
def test_solo_se_propone_escalar_lo_que_es_critico_abierto_y_vencido(cambios: dict[str, Any]) -> None:
    assert calcular_acciones(_estado_analizado(**cambios)) == []


def test_sin_analisis_no_hay_acciones() -> None:
    assert calcular_acciones(crear_estado(estado_final="completado")) == []


def test_si_no_hay_responsable_se_escala_al_comite() -> None:
    acciones = calcular_acciones(_estado_analizado(responsable=None))
    assert acciones[0].destinatario == "Comité de Gobierno de Datos"


def test_las_aristas_de_decision_usan_solo_los_destinos_previstos() -> None:
    assert decidir_tras_proponer(crear_estado(acciones_propuestas=[])) == "__end__"
    assert decidir_tras_proponer(crear_estado(acciones_propuestas=calcular_acciones(_estado_analizado()))) == NODO_APROBACION
    assert decidir_tras_aprobar(crear_estado(aprobacion=None)) == "__end__"
    assert decidir_tras_aprobar(crear_estado(aprobacion=DecisionHumana(aprobada=False, revisor="x"))) == "__end__"
    assert decidir_tras_aprobar(crear_estado(aprobacion=DecisionHumana(aprobada=True, revisor="x"))) == NODO_EJECUTAR


# ---------- La pausa ----------


async def test_una_tarea_critica_se_pausa_y_no_ejecuta_nada(
    armar_grafo_produccion: Callable[..., CompiledStateGraph], espia: NotificadorEspia
) -> None:
    grafo = armar_grafo_produccion(_supervisor(), _especialistas_criticos())
    await grafo.ainvoke(estado_inicial("tarea"), _hilo())
    foto = await grafo.aget_state(_hilo())
    assert foto.next == (NODO_APROBACION,)  # detenido, esperando una senal externa
    pausa = foto.interrupts[0].value
    assert pausa["tipo"] == "aprobacion_requerida"
    assert [a["incidente_id"] for a in pausa["acciones"]] == ["INC-1001", "INC-1002"]
    assert "Informe de calidad de datos" in pausa["informe"]
    assert espia.llamadas == []  # el efecto secundario NO ocurrio
    assert foto.values["acciones_ejecutadas"] == []


async def test_aprobada_la_pausa_se_ejecutan_las_acciones_y_termina(
    armar_grafo_produccion: Callable[..., CompiledStateGraph], espia: NotificadorEspia
) -> None:
    grafo = armar_grafo_produccion(_supervisor(), _especialistas_criticos())
    await grafo.ainvoke(estado_inicial("tarea"), _hilo("job-ok"))
    await grafo.ainvoke(Command(resume=APROBADA), _hilo("job-ok"))
    foto = await grafo.aget_state(_hilo("job-ok"))
    assert foto.next == ()  # termino
    assert foto.values["aprobacion"].revisor == "Cris"
    assert [r.incidente_id for r in foto.values["acciones_ejecutadas"]] == ["INC-1001", "INC-1002"]
    assert espia.llamadas == [("job-ok", "INC-1001"), ("job-ok", "INC-1002")]  # el hilo identifica el job
    assert foto.values["estado_final"] == "completado"


async def test_rechazada_la_pausa_no_se_ejecuta_nada(
    armar_grafo_produccion: Callable[..., CompiledStateGraph], espia: NotificadorEspia
) -> None:
    grafo = armar_grafo_produccion(_supervisor(), _especialistas_criticos())
    await grafo.ainvoke(estado_inicial("tarea"), _hilo())
    await grafo.ainvoke(Command(resume=RECHAZADA), _hilo())
    foto = await grafo.aget_state(_hilo())
    assert foto.next == ()
    assert foto.values["aprobacion"].aprobada is False
    assert foto.values["acciones_ejecutadas"] == []
    assert espia.llamadas == []
    assert foto.values["respuesta_final"]  # el informe se entrega igual


@pytest.mark.parametrize("senal", ["si", {"aprobada": "quizas", "revisor": "x"}, {"aprobada": True}, {"revisor": "x"}])
async def test_una_senal_ilegible_cuenta_como_rechazo(
    senal: Any, armar_grafo_produccion: Callable[..., CompiledStateGraph], espia: NotificadorEspia
) -> None:
    grafo = armar_grafo_produccion(_supervisor(), _especialistas_criticos())
    await grafo.ainvoke(estado_inicial("tarea"), _hilo())
    await grafo.ainvoke(Command(resume=senal), _hilo())
    foto = await grafo.aget_state(_hilo())
    assert foto.values["aprobacion"].aprobada is False
    assert foto.values["aprobacion"].revisor == "sistema"
    assert espia.llamadas == []


async def test_una_tarea_sin_incidentes_criticos_termina_sin_pausa(
    armar_grafo_produccion: Callable[..., CompiledStateGraph], espia: NotificadorEspia
) -> None:
    grafo = armar_grafo_produccion(_supervisor(), _especialistas_sin_criticos())
    await grafo.ainvoke(estado_inicial("tarea"), _hilo())
    foto = await grafo.aget_state(_hilo())
    assert foto.next == ()
    assert foto.values["acciones_propuestas"] == []
    assert foto.values["aprobacion"] is None
    assert espia.llamadas == []


async def test_otro_worker_puede_reanudar_el_job_con_el_checkpoint_compartido(
    armar_grafo_produccion: Callable[..., CompiledStateGraph], espia: NotificadorEspia
) -> None:
    """Simula dos procesos distintos: uno pausa el job y otro (grafo nuevo, mismo checkpointer) lo retoma."""
    almacen_de_checkpoints = InMemorySaver()
    primero = armar_grafo_produccion(_supervisor(), _especialistas_criticos(), checkpointer=almacen_de_checkpoints)
    await primero.ainvoke(estado_inicial("tarea"), _hilo("job-x"))
    segundo = armar_grafo_produccion(_supervisor(), especialistas_falsos(), checkpointer=almacen_de_checkpoints)
    await segundo.ainvoke(Command(resume=APROBADA), _hilo("job-x"))
    foto = await segundo.aget_state(_hilo("job-x"))
    assert foto.next == ()
    assert len(foto.values["acciones_ejecutadas"]) == 2
    # y lo que recupero el segundo grafo son los modelos de verdad, no diccionarios
    assert foto.values["analisis"].ranking == ["INC-1001", "INC-1002"]


# ---------- El nodo con efecto secundario se protege solo ----------


async def test_el_nodo_ejecutar_no_actua_sin_aprobacion_aunque_se_lo_invoque() -> None:
    espia = NotificadorEspia()
    nodo = crear_nodo_ejecutar(espia)
    propuestas = calcular_acciones(_estado_analizado())
    config = {"configurable": {"thread_id": "j"}}
    assert await nodo(crear_estado(acciones_propuestas=propuestas, aprobacion=None), config) == {"acciones_ejecutadas": []}  # type: ignore[arg-type]
    rechazo = DecisionHumana(aprobada=False, revisor="x")
    assert await nodo(crear_estado(acciones_propuestas=propuestas, aprobacion=rechazo), config) == {"acciones_ejecutadas": []}  # type: ignore[arg-type]
    assert espia.llamadas == []


async def test_si_una_accion_falla_las_demas_se_ejecutan_y_el_fallo_queda_registrado() -> None:
    espia = NotificadorEspia(fallar_en={"INC-1001"})
    nodo = crear_nodo_ejecutar(espia)
    estado = _estado_analizado()
    propuesta = calcular_acciones(estado)[0]
    otra = propuesta.model_copy(update={"incidente_id": "INC-1002"})
    resultado = await nodo(
        crear_estado(
            acciones_propuestas=[propuesta.model_copy(update={"incidente_id": "INC-1001"}), otra],
            aprobacion=DecisionHumana(aprobada=True, revisor="x"),
        ),
        {"configurable": {"thread_id": "j"}},  # type: ignore[arg-type]
    )
    ejecutadas = {r.incidente_id: r for r in resultado["acciones_ejecutadas"]}
    assert ejecutadas["INC-1001"].ejecutada is False
    assert "ConnectionError" in ejecutadas["INC-1001"].detalle
    assert ejecutadas["INC-1002"].ejecutada is True
