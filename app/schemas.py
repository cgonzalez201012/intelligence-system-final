"""Contratos de la API: todo lo que entra y sale por HTTP, validado con Pydantic."""

import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from orquestador.state import AccionPropuesta, DecisionHumana, EstadoFinal, ResultadoAccion

# Ciclo de vida de un job. PENDING = en la cola esperando a un worker (tambien despues de aprobar,
# mientras espera que un worker retome el grafo). WAITING_APPROVAL = el grafo esta pausado.
EstadoJob = Literal["PENDING", "RUNNING", "WAITING_APPROVAL", "DONE", "FAILED"]
ESTADOS_FINALES: frozenset[str] = frozenset({"DONE", "FAILED"})



class AprobacionIn(DecisionHumana):
    """Cuerpo de POST /tasks/{id}/approve: el mismo contrato que usa el grafo, sin campos de más y con ejemplo."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={"example": {"aprobada": True, "revisor": "Cris", "comentario": "Escalar a los responsables."}},
    )


class ErrorApi(BaseModel):
    """Forma de todos los errores que devuelve la API (404, 409, 503...)."""

    detail: str = Field(description="Qué pasó y, cuando se puede, qué hacer.")


class Salud(BaseModel):
    """Respuesta de GET /health."""

    estado: Literal["ok"]
    redis: Literal["ok"]
    tareas_en_cola: int = Field(ge=0)


class AprobacionPendiente(BaseModel):
    """Lo que ve la persona que tiene que decidir cuando un job queda en WAITING_APPROVAL."""

    tipo: Literal["aprobacion_requerida"]
    motivo: str
    acciones: list[AccionPropuesta]
    informe: str = Field(description="Informe del equipo (recortado) para decidir con contexto.")


class TareaIn(BaseModel):
    """Cuerpo de POST /tasks."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"tarea": "Priorizá los incidentes abiertos del maestro de artículos y citá la política que fija sus plazos."}},
    )

    tarea: str = Field(min_length=10, max_length=2000, description="Consulta para el equipo de agentes.")

    @field_validator("tarea")
    @classmethod
    def _sin_espacios_de_mas(cls, valor: str) -> str:
        if any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in valor):
            raise ValueError("La tarea no puede tener caracteres de control.")
        limpio = " ".join(valor.split())
        if len(limpio) < 10:
            raise ValueError("La tarea tiene que tener al menos 10 caracteres.")
        return limpio


class TareaCreada(BaseModel):
    """Respuesta inmediata de POST /tasks: el ticket para consultar el estado."""

    job_id: str
    estado: EstadoJob
    consultar_en: str


class UsoDelModelo(BaseModel):
    """Llamadas y tokens gastados por el job (el costo en dólares lo calcula el dashboard de trazas)."""

    llamadas_al_modelo: int = 0
    tokens_entrada: int = 0
    tokens_salida: int = 0

    def sumar(self, otro: "UsoDelModelo") -> "UsoDelModelo":
        return UsoDelModelo(
            llamadas_al_modelo=self.llamadas_al_modelo + otro.llamadas_al_modelo,
            tokens_entrada=self.tokens_entrada + otro.tokens_entrada,
            tokens_salida=self.tokens_salida + otro.tokens_salida,
        )


class AccionesDelJob(BaseModel):
    """Lo que pasó con las acciones críticas: qué se propuso, qué decidió la persona y qué se ejecutó."""

    propuestas: list[AccionPropuesta] = Field(default_factory=list)
    decision: DecisionHumana | None = None
    ejecutadas: list[ResultadoAccion] = Field(default_factory=list)


class ResultadoTarea(BaseModel):
    """Resultado de un job terminado."""

    estado_final: EstadoFinal | Literal["desconocido"] = Field(
        description="completado, limite_de_pasos o validacion_fallida (lo decide el supervisor)."
    )
    respuesta: str
    fuentes_citadas: list[str] = Field(
        default_factory=list,
        description="Políticas del RAG que la respuesta cita, con su código y fuente (por ejemplo 'GD-POL-001 (01-politica.md, pág. 1)').",
    )
    agentes_que_intervinieron: list[str] = Field(default_factory=list)
    delegaciones: int = 0
    acciones: AccionesDelJob = Field(default_factory=AccionesDelJob)


class TareaEstado(BaseModel):
    """Respuesta de GET /tasks/{id}: el registro del job en Redis."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    estado: EstadoJob
    tarea: str
    creado_en: str
    actualizado_en: str
    finalizado_en: str | None = None
    resultado: ResultadoTarea | None = None
    error: str | None = None
    aprobacion_pendiente: AprobacionPendiente | None = None
    decision: DecisionHumana | None = None
    uso: UsoDelModelo = Field(default_factory=UsoDelModelo)
    segundos_de_ejecucion: float = 0.0
    recuperaciones: int = Field(
        default=0, description="Veces que otro worker retomó este job desde su checkpoint porque el anterior se cayó."
    )


class MensajeDeCola(BaseModel):
    """Lo que viaja por la cola de Redis entre la API y el worker."""

    job_id: str
    accion: Literal["ejecutar", "reanudar"]
