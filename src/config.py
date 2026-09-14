"""
Configuration centrale du projet.

Toutes les valeurs sont surchargeables via variables d'environnement (.env),
ce qui permet de faire tourner le même code :
  - en local avec un modèle mock (aucune clé API requise, pour développer/tester)
  - avec l'API OpenAI
  - avec Azure OpenAI Service
  - avec un modèle local servi par Ollama

Voir .env.example pour la liste complète des variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv est optionnel : si absent, on lit directement os.environ
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"


@dataclass
class LLMConfig:
    """Configuration du fournisseur LLM utilisé par le RAG et le scoring.

    provider: "mock" | "openai" | "azure_openai" | "ollama"
    """
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock"))

    # OpenAI
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    # Azure OpenAI (mêmes briques que celles utilisées par VINCI Construction SI
    # dans l'offre de stage : Azure OpenAI Service)
    azure_api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    azure_endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    azure_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT", ""))
    azure_api_version: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"))

    # Ollama (LLM local, ex. llama3, mistral)
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3"))

    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))


@dataclass
class EmbeddingConfig:
    """Configuration du modèle d'embeddings utilisé pour l'indexation RAG.

    Par défaut : sentence-transformers local (gratuit, pas de clé API, tourne en CPU),
    ce qui garantit que le projet fonctionne "out of the box".
    """
    provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "tfidf"))
    local_model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    )
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )


@dataclass
class RAGConfig:
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "4")))
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.25"))
    )
    collection_name: str = "governance_knowledge_base"


llm_config = LLMConfig()
embedding_config = EmbeddingConfig()
rag_config = RAGConfig()
