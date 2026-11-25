"""Proveedor compartido del RAG: se arma una sola vez por proceso y se reutiliza en todas las consultas.

Armar el recuperador cuesta (abrir el indice de Pinecone, crear el cliente de embeddings y bajar el corpus
para BM25), asi que no se hace por cada llamada a la herramienta. Es perezoso: si nadie consulta politicas,
no se toca la red ni se piden las claves de Pinecone y Gemini.
"""

import asyncio

from dotenv import load_dotenv

from rag import config as rag_config
from rag.rag_system import RAGSystem

_rag: RAGSystem | None = None
_candado: asyncio.Lock | None = None


async def obtener_rag() -> RAGSystem:
    """Devuelve el RAGSystem compartido, creandolo en la primera llamada (seguro ante llamadas concurrentes).

    Lanza `rag.errores.ConfiguracionError` si faltan variables de entorno y `IndiceError` / `NamespaceVacioError`
    si el indice no existe o esta vacio: la herramienta los traduce a un mensaje con instrucciones.
    """
    global _rag, _candado
    if _rag is not None:
        return _rag
    if _candado is None:
        _candado = asyncio.Lock()
    async with _candado:
        if _rag is None:
            load_dotenv()
            cfg = rag_config.cargar_configuracion()
            _rag = await RAGSystem.desde_configuracion(cfg)
    return _rag


def fijar_rag(rag: RAGSystem | None) -> None:
    """Inyecta un RAG ya armado (pruebas) o lo descarta (`None`) para forzar una nueva creacion."""
    global _rag, _candado
    _rag = rag
    _candado = None
