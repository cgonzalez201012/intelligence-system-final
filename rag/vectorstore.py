"""Vector store de Pinecone seguro para muchas busquedas simultaneas dentro de un event loop.

`PineconeVectorStore.asimilarity_search` (langchain-pinecone) usa un cliente asincrono (aiohttp) que comparte su
sesion entre llamadas: con dos busquedas a la vez, la primera en terminar cierra la sesion mientras la otra sigue en
vuelo ("Session is closed"). Con un worker que atiende varios jobs concurrentes eso pasaria en produccion.

Aca la busqueda asincrona delega en la sincrona, que se ejecuta en un hilo (`asyncio.to_thread`): el event loop nunca
se bloquea y cada busqueda usa su propia llamada HTTP, sin estado compartido. El cliente sincrono del SDK de Pinecone
admite llamadas desde varios hilos.
"""

import asyncio
from typing import Any

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore


class PineconeVectorStoreEnHilo(PineconeVectorStore):
    """`PineconeVectorStore` cuya busqueda asincrona corre la sincrona en un hilo."""

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        namespace: str | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return await asyncio.to_thread(
            self.similarity_search, query, k=k, filter=filter, namespace=namespace, **kwargs
        )
