"""Recuperador hibrido: BM25 (lexico) + Pinecone (semantico) fusionados con un EnsembleRetriever.

Uso desde codigo (todo asincrono: ninguna llamada bloquea el event loop):
    rag = await RAGSystem.desde_configuracion(cfg)
    fragmentos = await rag.buscar("plazo de resolucion de un incidente critico")   # los 5 mejores

Por que hibrido (para defenderlo):
* La busqueda semantica (embeddings) entiende sinonimos y parafrasis, pero puede confundir codigos
  o nombres propios parecidos (GD-PRO-003 vs GD-PRO-004).
* BM25 acierta con terminos exactos (codigos, siglas, nombres), pero no entiende sinonimos.
* El EnsembleRetriever fusiona los dos rankings con Reciprocal Rank Fusion (RRF): suma pesos / (posicion + 60).
  Usa POSICIONES, no puntajes, porque los puntajes de BM25 y de coseno tienen escalas incomparables
  (error conceptual comun: sumar los scores a mano).

Detalles de implementacion:
* Asincronia: el SDK de Pinecone, BM25 y la libreria de embeddings son sincronos. Nada de eso corre en el event loop:
  la construccion (listar y leer el corpus) va en `asyncio.to_thread` y las busquedas usan `retriever.ainvoke`, que
  LangChain ejecuta en un hilo cuando el recuperador no tiene version asincrona propia.
* BM25 se arma en memoria con el texto que se guardo en la metadata de Pinecone (cargar_corpus).
* El EnsembleRetriever devuelve la union de ambas listas (hasta 2k): aca se recorta a los k mejores.
* El tokenizador de BM25 normaliza mayusculas y tildes y conserva los codigos con guiones ("gd-pro-003")
  como un solo termino. El de LangChain por defecto solo parte por espacios (no normaliza nada).
* Si ninguna palabra de la consulta aparece en un fragmento, BM25 no aporta ese fragmento (puntaje 0 =
  sin evidencia lexica): asi no mete ruido en la fusion cuando la pregunta esta parafraseada.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from rag import config
from rag.almacen import abrir_indice, cargar_corpus, crear_cliente
from rag.embeddings import crear_embeddings, es_error_de_cuota
from rag.errores import EmbeddingsError, FiltroSinResultadosError, NamespaceVacioError
from rag.schemas import ConsultaRAG, FragmentoRecuperado, ModoBusqueda, ResultadoRAG

logger = logging.getLogger("rag.sistema")

MODOS = ("bm25", "vectorial", "hibrido")   # el sistema final va al ultimo

# Reintentos de la consulta ante cuota agotada de Gemini (429): espera creciente, sin bloquear el loop.
BUSQUEDA_MAX_INTENTOS = 3
BUSQUEDA_ESPERA_INICIAL_S = 2.0

_TERMINO = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# Palabras gramaticales del espanol (ya sin tildes). BM25 las ignora: aparecen en casi todos los fragmentos, no
# distinguen a ninguno, y sin este filtro una consulta como "¿Que es Depo360?" traia fragmentos al azar solo porque
# contenian "que" o "es". Las palabras con contenido (dias, plazo, proveedor, ...) y los numeros se conservan.
STOPWORDS_ES = frozenset(
    """a al algo ante bajo como con contra cual cuales cuando cuanto cuanta cuantos cuantas de del desde donde
    durante e el ella ellas ellos en entre era es esa ese eso esta estan este esto estos estas fue ha han hasta
    hay la las le les lo los mas me mi mis muy ni o para pero por que quien quienes se segun ser si sido sobre
    son su sus tambien te tras tu un una uno unos unas y ya""".split()
)


def normalizar_para_bm25(texto: str) -> list[str]:
    """Minusculas, sin tildes, sin palabras vacias, y terminos alfanumericos.

    Los guiones internos se conservan para que un codigo sea UN solo termino: 'GD-PRO-003' -> 'gd-pro-003'.
    """
    sin_tildes = "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")
    return [t for t in _TERMINO.findall(sin_tildes) if t not in STOPWORDS_ES]


class BM25SoloCoincidencias(BM25Retriever):
    """BM25Retriever que descarta los fragmentos con puntaje 0 (ninguna palabra de la consulta aparece en ellos)."""

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        puntajes = self.vectorizer.get_scores(self.preprocess_func(query))
        orden = sorted(range(len(self.docs)), key=lambda i: puntajes[i], reverse=True)
        return [self.docs[i] for i in orden[: self.k] if puntajes[i] > 0]


def _a_fragmento(posicion: int, doc: Document) -> FragmentoRecuperado:
    """Convierte un Document de LangChain en el modelo Pydantic que sale del modulo."""
    m = doc.metadata
    return FragmentoRecuperado(
        posicion=posicion,
        documento_id=str(m.get("documento_id", "")),
        fuente=str(m.get("fuente", "")),
        pagina=int(m.get("pagina", 0) or 0),
        seccion=str(m.get("seccion", "")),
        categoria=str(m.get("categoria", "")),
        codigo=str(m.get("codigo", "")),
        chunk_id=str(m.get("chunk_id", "")),
        texto=doc.page_content,
    )


@dataclass(frozen=True)
class _Recuperadores:
    bm25: BM25Retriever
    vectorial: Any
    ensemble: EnsembleRetriever


class RAGSystem:
    """Encapsula el EnsembleRetriever (BM25 + vectorial). Recibe una consulta y devuelve los top-k fragmentos."""

    def __init__(
        self,
        vectorstore: Any,
        documentos: list[Document],
        *,
        k: int = config.TOP_K,
        pesos: tuple[float, float] = config.PESOS_HIBRIDO,
        candidatos: int | None = None,
    ) -> None:
        if not documentos:
            raise NamespaceVacioError("No hay documentos para armar el recuperador. Indexa primero: python -m rag.ingest")
        if k < 1:
            raise ValueError("k debe ser >= 1")
        if len(pesos) != 2 or min(pesos) < 0 or sum(pesos) <= 0:
            raise ValueError("pesos debe ser (peso_bm25, peso_vectorial), sin negativos y con suma > 0")
        if candidatos is not None and candidatos < k:
            raise ValueError("candidatos debe ser >= k (cada recuperador aporta al menos k fragmentos a la fusion)")
        self.k = k
        # Cuantos fragmentos aporta CADA recuperador a la fusion antes de recortar a los k finales.
        self.candidatos = candidatos if candidatos is not None else max(k, config.CANDIDATOS_POR_RECUPERADOR)
        self.pesos = (float(pesos[0]), float(pesos[1]))
        self._vectorstore = vectorstore
        self._documentos = list(documentos)
        self._cache: dict[str | None, _Recuperadores] = {}
        # El recuperador hibrido "oficial" (sin filtro de categoria):
        self.ensemble: EnsembleRetriever = self._recuperadores(None).ensemble

    # ------------------------------------------------------------------ construccion
    @classmethod
    async def desde_indice(
        cls,
        index: Any,
        embeddings: Any,
        namespace: str,
        *,
        vectorstore_factory: Callable[..., Any] | None = None,
        **opciones: Any,
    ) -> "RAGSystem":
        """Arma el sistema sobre un indice de Pinecone ya abierto (el texto de BM25 se lee de la metadata).

        Listar y leer el corpus son llamadas sincronas al SDK: van en un hilo para no frenar el event loop.
        """
        if vectorstore_factory is None:
            from langchain_pinecone import PineconeVectorStore

            vectorstore_factory = PineconeVectorStore
        vectorstore = vectorstore_factory(
            index=index, embedding=embeddings, text_key=config.TEXT_KEY, namespace=namespace
        )
        corpus = await asyncio.to_thread(cargar_corpus, index, namespace)
        if not corpus:
            raise NamespaceVacioError(
                f"El namespace '{namespace}' no tiene fragmentos. Indexa primero: python -m rag.ingest "
                "(si lo acabas de correr, espera unos segundos: Pinecone es eventualmente consistente)."
            )
        return cls(vectorstore, corpus, **opciones)

    @classmethod
    async def desde_configuracion(cls, cfg: config.Configuracion, **opciones: Any) -> "RAGSystem":
        def _abrir() -> Any:
            return abrir_indice(crear_cliente(cfg), cfg.index_name)

        index = await asyncio.to_thread(_abrir)
        embeddings = await asyncio.to_thread(crear_embeddings, cfg)
        return await cls.desde_indice(index, embeddings, cfg.namespace, **opciones)

    def _recuperadores(self, categoria: str | None) -> _Recuperadores:
        if categoria in self._cache:
            return self._cache[categoria]

        if categoria is None:
            documentos = self._documentos
        else:
            documentos = [d for d in self._documentos if d.metadata.get("categoria") == categoria]
            if not documentos:
                disponibles = sorted({str(d.metadata.get("categoria")) for d in self._documentos})
                raise FiltroSinResultadosError(
                    f"No hay documentos de la categoria '{categoria}'. Categorias disponibles: {disponibles}."
                )

        bm25 = BM25SoloCoincidencias.from_documents(documentos, k=self.candidatos, preprocess_func=normalizar_para_bm25)
        opciones_busqueda: dict[str, Any] = {"k": self.candidatos}
        if categoria is not None:
            # Filtro por metadatos: Pinecone descarta ANTES de comparar vectores (no despues).
            opciones_busqueda["filter"] = {"categoria": {"$eq": categoria}}
        vectorial = self._vectorstore.as_retriever(search_kwargs=opciones_busqueda)
        ensemble = EnsembleRetriever(
            retrievers=[bm25, vectorial],
            weights=list(self.pesos),
            id_key=config.CAMPO_ID_FUSION,
        )
        self._cache[categoria] = _Recuperadores(bm25, vectorial, ensemble)
        return self._cache[categoria]

    # ------------------------------------------------------------------ busqueda
    async def buscar(
        self, consulta: str, *, categoria: str | None = None, modo: ModoBusqueda = "hibrido", k: int | None = None
    ) -> list[FragmentoRecuperado]:
        """Devuelve hasta `k` fragmentos ordenados de mejor a peor.

        modo: "hibrido" (BM25 + vectorial, el sistema), o "vectorial" / "bm25" por separado (para comparar).
        categoria: filtra por la metadata `categoria` (por ejemplo "politica"). Si Gemini devuelve 429 al
        convertir la pregunta en vector, se reintenta con espera creciente (asyncio.sleep, sin bloquear).
        """
        consulta = (consulta or "").strip()
        if not consulta:
            raise ValueError("La consulta esta vacia.")
        if modo not in MODOS:
            raise ValueError(f"modo debe ser uno de {MODOS}, no '{modo}'.")
        limite = self.k if k is None else k
        if limite < 1:
            raise ValueError("k debe ser >= 1")

        if categoria not in self._cache:
            # Armar el BM25 de una categoria es trabajo de CPU: se hace una vez, en un hilo, y queda en cache.
            await asyncio.to_thread(self._recuperadores, categoria)
        recuperadores = self._recuperadores(categoria)
        retriever = {
            "hibrido": recuperadores.ensemble,
            "vectorial": recuperadores.vectorial,
            "bm25": recuperadores.bm25,
        }[modo]

        espera = BUSQUEDA_ESPERA_INICIAL_S
        for intento in range(1, BUSQUEDA_MAX_INTENTOS + 1):
            try:
                documentos = await retriever.ainvoke(consulta)
                break
            except Exception as exc:  # noqa: BLE001 - solo se reintenta la cuota agotada
                if not es_error_de_cuota(exc):
                    raise
                if intento == BUSQUEDA_MAX_INTENTOS:
                    raise EmbeddingsError(
                        f"Gemini siguio limitando el uso tras {BUSQUEDA_MAX_INTENTOS} intentos al buscar. "
                        "Es la cuota gratis: espera un minuto y volve a intentar."
                    ) from exc
                logger.warning("Limite de uso al buscar (intento %s/%s). Espero %.0f s...", intento, BUSQUEDA_MAX_INTENTOS, espera)
                await asyncio.sleep(espera)
                espera *= 2
        return [_a_fragmento(i, d) for i, d in enumerate(documentos[:limite], start=1)]

    async def consultar(self, consulta: ConsultaRAG) -> ResultadoRAG:
        """Version tipada de `buscar`: entra una ConsultaRAG validada, sale un ResultadoRAG con la latencia medida."""
        inicio = time.perf_counter()
        fragmentos = await self.buscar(consulta.consulta, categoria=consulta.categoria, modo=consulta.modo, k=consulta.k)
        return ResultadoRAG(
            consulta=consulta.consulta,
            categoria=consulta.categoria,
            modo=consulta.modo,
            fragmentos=fragmentos,
            latencia_ms=round((time.perf_counter() - inicio) * 1000, 1),
        )

    @property
    def categorias(self) -> list[str]:
        return sorted({str(d.metadata.get("categoria")) for d in self._documentos})

    @property
    def documentos_indexados(self) -> list[str]:
        return sorted({str(d.metadata.get("documento_id")) for d in self._documentos})
