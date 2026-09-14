"""
Retriever : interroge le vector store et structure les résultats en
"passages sourcés" exploitables par le chatbot et par le générateur de
rapports.

Point de conformité important : si aucun passage ne dépasse le seuil de
similarité configuré (rag_config.similarity_threshold), le retriever renvoie
une liste vide plutôt que de forcer des résultats non pertinents — cela évite
au chatbot d'halluciner une réponse sourcée sur du contenu hors-sujet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.config import rag_config
from src.rag.vectorstore import VectorStore


@dataclass
class RetrievedPassage:
    chunk_id: str
    text: str
    referentiel: str
    domaine: str
    titre: str
    similarity: float

    def as_citation(self) -> str:
        return f"{self.chunk_id} ({self.referentiel} — {self.titre})"


class Retriever:
    def __init__(self, vectorstore: VectorStore):
        self._vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedPassage]:
        raw_results = self._vectorstore.search(query, top_k=top_k)
        passages: List[RetrievedPassage] = []
        for chunk_id, text, metadata, similarity in raw_results:
            if similarity < rag_config.similarity_threshold:
                continue
            passages.append(
                RetrievedPassage(
                    chunk_id=chunk_id,
                    text=text,
                    referentiel=metadata.get("referentiel", "inconnu"),
                    domaine=metadata.get("domaine", "inconnu"),
                    titre=metadata.get("titre", ""),
                    similarity=round(similarity, 3),
                )
            )
        return passages
