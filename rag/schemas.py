"""Contratos Pydantic del RAG: lo que entra a una consulta y lo que vuelve de ella.

Todo lo que cruza el borde del módulo (la herramienta del investigador, los tests) pasa por estos modelos:
si una consulta viene vacía o con un `k` absurdo, falla acá con un mensaje claro y no dentro de Pinecone.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CategoriaDocumento = Literal["politica", "roles", "procedimiento", "glosario", "general"]
ModoBusqueda = Literal["hibrido", "vectorial", "bm25"]


class ConsultaRAG(BaseModel):
    """Una pregunta a la base documental."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    consulta: str = Field(min_length=3, max_length=500, description="La pregunta o los términos a buscar.")
    categoria: CategoriaDocumento | None = Field(
        default=None, description="Filtra por el tipo de documento (metadata `categoria`)."
    )
    k: int = Field(default=5, ge=1, le=10, description="Cuántos fragmentos devolver como máximo.")
    modo: ModoBusqueda = Field(default="hibrido", description="Híbrido (BM25 + vectorial) o cada uno por separado.")


class FragmentoRecuperado(BaseModel):
    """Un fragmento de documento recuperado, con todo lo necesario para citarlo."""

    model_config = ConfigDict(frozen=True)

    posicion: int = Field(ge=1, description="1 = el más relevante.")
    documento_id: str
    fuente: str = Field(description="Archivo de origen, por ejemplo 01-politica-calidad-de-datos.md.")
    pagina: int = Field(ge=0)
    seccion: str = ""
    categoria: str = ""
    codigo: str = Field(default="", description="Código del documento, por ejemplo GD-POL-001.")
    chunk_id: str
    texto: str

    @property
    def cita(self) -> str:
        """Referencia corta y estable para citar el fragmento: `GD-POL-001 (01-politica...md, pág. 1)`."""
        codigo = self.codigo or self.documento_id
        return f"{codigo} ({self.fuente}, pág. {self.pagina})"


class ResultadoRAG(BaseModel):
    """Respuesta de una consulta: los fragmentos y cuánto tardó (para las trazas y la evidencia)."""

    model_config = ConfigDict(frozen=True)

    consulta: str
    categoria: str | None
    modo: ModoBusqueda
    fragmentos: list[FragmentoRecuperado]
    latencia_ms: float = Field(ge=0)
