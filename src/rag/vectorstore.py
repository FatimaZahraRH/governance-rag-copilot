"""
Wrapper autour de ChromaDB (base vectorielle locale, persistée sur disque).

Choix de ChromaDB plutôt qu'un service managé (Azure AI Search, Pinecone...) :
ChromaDB tourne en local sans dépendance externe, ce qui garantit que le
projet est démontrable immédiatement, sur n'importe quelle machine.

Pour une bascule vers Azure AI Search (cohérent avec l'écosystème Microsoft
mentionné dans l'offre VINCI Construction SI), il suffirait d'implémenter la
même interface (add_documents / search) avec le SDK azure-search-documents —
le reste du code (retriever, chatbot) n'aurait pas à changer.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from src.config import VECTORSTORE_DIR, rag_config
from src.ingestion.loader import KnowledgeChunk
from src.rag.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, embedding_provider: EmbeddingProvider, collection_name: Optional[str] = None):
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("Le package 'chromadb' n'est pas installé. Lancez : pip install chromadb") from exc

        self._embedding_provider = embedding_provider
        self._client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        self._collection_name = collection_name or rag_config.collection_name
        # get_or_create_collection : idempotent, pratique pour re-run le script d'indexation
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

    def reset(self) -> None:
        """Supprime et recrée la collection (utile pour ré-indexer proprement)."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(name=self._collection_name)

    def add_chunks(self, chunks: List[KnowledgeChunk]) -> None:
        if not chunks:
            logger.warning("Aucun chunk à indexer.")
            return
        texts = [c.text for c in chunks]
        embeddings = self._embedding_provider.embed_texts(texts)
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks],
        )
        logger.info("%d chunks indexés dans la collection '%s'.", len(chunks), self._collection_name)

    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[str, str, Dict, float]]:
        """Retourne une liste de tuples (chunk_id, texte, métadonnées, score_similarite).

        Le score retourné est une similarité (plus haut = plus pertinent),
        obtenue en convertissant la distance retournée par Chroma.
        """
        k = top_k or rag_config.top_k
        query_embedding = self._embedding_provider.embed_query(query)
        results = self._collection.query(query_embeddings=[query_embedding], n_results=k)

        output: List[Tuple[str, str, Dict, float]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
            # Chroma retourne une distance (cosine ou L2 selon config) ; on la
            # convertit en un score de similarité borné [0, 1] pour l'interprétation.
            similarity = max(0.0, 1.0 - distance)
            output.append((chunk_id, doc, meta, similarity))
        return output

    def count(self) -> int:
        return self._collection.count()
