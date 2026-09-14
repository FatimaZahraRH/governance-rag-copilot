"""
Fournisseur d'embeddings — même logique d'agnosticisme que llm_provider.py.

Par défaut on utilise un modèle local (sentence-transformers, multilingue),
ce qui permet d'indexer et d'interroger la base de connaissances sans clé API
et gratuitement. Un mode OpenAI est disponible si vous voulez des embeddings
de meilleure qualité en production.
"""
from __future__ import annotations

import abc
import logging
from typing import List, Optional

from src.config import EmbeddingConfig, embedding_config

logger = logging.getLogger(__name__)


class EmbeddingProvider(abc.ABC):
    @abc.abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


class LocalEmbeddingProvider(EmbeddingProvider):
    """Utilise sentence-transformers (modèle téléchargé une fois depuis Hugging
    Face, puis exécuté en local). Meilleure qualité sémantique, recommandé en
    conditions réelles (sur une machine avec accès Internet)."""

    def __init__(self, config: EmbeddingConfig):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Le package 'sentence-transformers' n'est pas installé. "
                "Lancez : pip install sentence-transformers"
            ) from exc
        logger.info("Chargement du modèle d'embeddings local : %s", config.local_model_name)
        self._model = SentenceTransformer(config.local_model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


class TfidfEmbeddingProvider(EmbeddingProvider):
    """Fournisseur d'embeddings 100% local et hors-ligne, basé sur un TF-IDF
    (scikit-learn) avec réduction dimensionnelle (SVD tronqué).

    Aucun téléchargement de modèle requis : c'est le mode par défaut le plus
    robuste (utile en environnement réseau restreint, en CI, ou pour une
    démonstration garantie de fonctionner sans dépendance externe).

    Limite assumée : la qualité sémantique est inférieure à un modèle de
    type sentence-transformers (pas de vraie compréhension du sens, seulement
    de la similarité lexicale pondérée). Pour la production, privilégiez
    EMBEDDING_PROVIDER=local (sentence-transformers) ou =openai.

    Le vocabulaire est appris (fit) une seule fois sur l'ensemble de la base
    de connaissances lors de l'indexation (embed_texts), puis PERSISTÉ SUR
    DISQUE (vectorizer + SVD) afin qu'une nouvelle exécution du programme
    (ex. l'API FastAPI lancée séparément du script d'indexation) puisse
    recharger le même espace vectoriel pour vectoriser les questions
    utilisateur (embed_query) de façon cohérente avec l'index déjà construit.
    """

    def __init__(self, n_components: int = 128, persist_path: "Path | None" = None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        from src.config import VECTORSTORE_DIR

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=4096,
        )
        self._n_components = n_components
        self._svd = None
        self._fitted = False
        self._persist_path = persist_path or (VECTORSTORE_DIR / "tfidf_embedder.joblib")
        self._try_load()

    def _try_load(self) -> None:
        if self._persist_path.exists():
            import joblib

            state = joblib.load(self._persist_path)
            self._vectorizer = state["vectorizer"]
            self._svd = state["svd"]
            self._fitted = True
            logger.info("Vectorizer TF-IDF rechargé depuis %s", self._persist_path)

    def _save(self) -> None:
        import joblib

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self._vectorizer, "svd": self._svd}, self._persist_path)
        logger.info("Vectorizer TF-IDF sauvegardé dans %s", self._persist_path)

    def _l2_normalize(self, matrix):
        import numpy as np

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        from sklearn.decomposition import TruncatedSVD

        tfidf_matrix = self._vectorizer.fit_transform(texts)
        # n_components doit être < nombre de documents pour TruncatedSVD
        n_components = min(self._n_components, max(2, tfidf_matrix.shape[0] - 1), tfidf_matrix.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = self._svd.fit_transform(tfidf_matrix)
        self._fitted = True
        self._save()
        return self._l2_normalize(reduced).tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self._fitted or self._svd is None:
            raise RuntimeError(
                "TfidfEmbeddingProvider n'a pas encore été 'fit' sur la base de connaissances. "
                "Lancez d'abord scripts/build_index.py pour indexer les documents."
            )
        tfidf_vector = self._vectorizer.transform([text])
        reduced = self._svd.transform(tfidf_vector)
        return self._l2_normalize(reduced)[0].tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingConfig, api_key: str):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Le package 'openai' n'est pas installé.") from exc
        if not api_key:
            raise ValueError("OPENAI_API_KEY manquant pour les embeddings OpenAI.")
        self._client = OpenAI(api_key=api_key)
        self._model = config.openai_embedding_model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


def get_embedding_provider(config: Optional[EmbeddingConfig] = None, openai_api_key: str = "") -> EmbeddingProvider:
    cfg = config or embedding_config
    if cfg.provider == "local":
        return LocalEmbeddingProvider(cfg)
    if cfg.provider == "tfidf":
        return TfidfEmbeddingProvider()
    if cfg.provider == "openai":
        return OpenAIEmbeddingProvider(cfg, openai_api_key)
    raise ValueError(f"Fournisseur d'embeddings inconnu : '{cfg.provider}'")
