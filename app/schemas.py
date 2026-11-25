"""Contratos de la API: todo lo que entra y sale por HTTP, validado con Pydantic."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from orquestador.state import AccionPropuesta, DecisionHumana, ResultadoAccion

# Ciclo de vida de un job. PENDING = en la cola esperando a un worker (tambien despues de aprobar,
# mientras espera que un worker retome el grafo). WAITING_APPROVAL = el grafo esta pausado.
EstadoJob = Literal["PENDING", "RUNNING", "WAITING_APPROVAL", "DONE", "FAILED"]
ESTADOS_FINALES: frozenset[str] = frozenset({"DONE", "FAILED"})

# La decision humana entra por el endpoint /approve con el mismo contrato que usa el grafo.
AprobacionIn = DecisionHumana


class TareaIn(BaseModel):
    """Cuerpo de POST /tasks."""

    tarea: str = Field(min_length=10, max_length=2000, description="Consulta para el equipo de agentes.")

    @field_validator("tarea")
    @classmethod
    def _sin_espacios_de_mas(cls, valor: str) -> str:
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
    """Llamadas y tokens gastados por el job (el costo en dolares lo calcula el dashboard de trazas)."""

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
    """Lo que paso con las acciones criticas: que se propuso, que decidio la persona y que se ejecuto."""

    propuestas: list[AccionPropuesta] = Field(default_factory=list)
    decision: DecisionHumana | None = None
    ejecutadas: list[ResultadoAccion] = Field(default_factory=list)


class ResultadoTarea(BaseModel):
    """Resultado de un job terminado."""

    estado_final: str = Field(description="completado, limite_de_pasos o validacion_fallida (lo decide el supervisor).")
    respuesta: str
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
    aprobacion_pendiente: dict[str, Any] | None = None
    decision: DecisionHumana | None = None
    uso: UsoDelModelo = Field(default_factory=UsoDelModelo)
    segundos_de_ejecucion: float = 0.0


class MensajeDeCola(BaseModel):
    """Lo que viaja por la cola de Redis entre la API y el worker."""

    job_id: str
    accion: Literal["ejecutar", "reanudar"]
