"""
Couche d'abstraction LLM — cœur de l'architecture agnostique du projet.

Objectif pédagogique : montrer comment concevoir un système RAG qui ne dépend
d'AUCUN fournisseur LLM particulier. Chaque fournisseur implémente la même
interface abstraite (LLMProvider), ce qui permet de basculer d'OpenAI à
Azure OpenAI ou à un modèle local en changeant une seule variable
d'environnement (LLM_PROVIDER), sans toucher au reste du code (RAG, scoring,
API...).

C'est un pattern "Strategy" classique, très utilisé en ingénierie logicielle
pour éviter le "vendor lock-in" — un point souvent valorisé en entretien
technique.
"""
from __future__ import annotations

import abc
import logging
from typing import Optional

from src.config import LLMConfig, llm_config

logger = logging.getLogger(__name__)


class LLMProvider(abc.ABC):
    """Interface commune que tout fournisseur LLM doit implémenter."""

    @abc.abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> str:
        """Génère une réponse texte à partir d'un prompt système + utilisateur."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    """Fournisseur factice, sans dépendance externe ni clé API.

    Utile pour :
      - développer/tester le pipeline RAG et le moteur de scoring sans compte payant,
      - faire tourner les tests unitaires en CI sans secrets,
      - faire une démonstration hors-ligne.

    Il ne fait PAS d'inférence réelle : il compose une réponse déterministe et
    transparente à partir du contexte fourni, en particulier des extraits
    récupérés par le RAG, pour que la démo reste crédible et vérifiable.
    """

    @property
    def name(self) -> str:
        return "mock"

    def generate(self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> str:
        logger.info("[MockLLMProvider] Génération factice (aucun appel externe)")
        return (
            "[Réponse générée en mode MOCK — aucun LLM réel n'a été appelé]\n\n"
            "Voici une synthèse basée strictement sur le contexte fourni ci-dessous. "
            "Pour obtenir une vraie génération en langage naturel, configurez "
            "LLM_PROVIDER=openai (ou azure_openai / ollama) et la clé associée dans .env.\n\n"
            f"--- Contexte transmis au modèle ---\n{user_prompt[:1200]}"
        )


class OpenAIProvider(LLMProvider):
    """Fournisseur OpenAI (API standard, api.openai.com)."""

    def __init__(self, config: LLMConfig):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Le package 'openai' n'est pas installé. Lancez : pip install openai"
            ) from exc
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY manquant dans l'environnement (.env).")
        self._client = OpenAI(api_key=config.openai_api_key)
        self._model = config.openai_model
        self._config = config

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def generate(self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature if temperature is not None else self._config.temperature,
        )
        return response.choices[0].message.content or ""


class AzureOpenAIProvider(LLMProvider):
    """Fournisseur Azure OpenAI Service.

    Choix déterminant pour une candidature comme celle de VINCI Construction SI,
    qui mentionne explicitement Azure OpenAI / Copilot Studio dans son stack :
    ce provider permet de dire en entretien "mon architecture est directement
    compatible avec votre écosystème Microsoft, il suffit de configurer les
    identifiants Azure".
    """

    def __init__(self, config: LLMConfig):
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise ImportError(
                "Le package 'openai' n'est pas installé. Lancez : pip install openai"
            ) from exc
        if not (config.azure_api_key and config.azure_endpoint and config.azure_deployment):
            raise ValueError(
                "Configuration Azure OpenAI incomplète : renseignez AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_ENDPOINT et AZURE_OPENAI_DEPLOYMENT dans .env."
            )
        self._client = AzureOpenAI(
            api_key=config.azure_api_key,
            azure_endpoint=config.azure_endpoint,
            api_version=config.azure_api_version,
        )
        self._deployment = config.azure_deployment
        self._config = config

    @property
    def name(self) -> str:
        return f"azure_openai:{self._deployment}"

    def generate(self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> str:
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature if temperature is not None else self._config.temperature,
        )
        return response.choices[0].message.content or ""


class OllamaProvider(LLMProvider):
    """Fournisseur pour un LLM local servi par Ollama (ex. llama3, mistral).

    Utile pour une démonstration 100% locale, sans donnée envoyée à un tiers —
    argument pertinent dans un contexte de gouvernance/confidentialité.
    """

    def __init__(self, config: LLMConfig):
        try:
            import requests  # noqa: F401
        except ImportError as exc:
            raise ImportError("Le package 'requests' n'est pas installé.") from exc
        self._base_url = config.ollama_base_url
        self._model = config.ollama_model
        self._config = config

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def generate(self, system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> str:
        import requests

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature if temperature is not None else self._config.temperature},
        }
        resp = requests.post(f"{self._base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")


def get_llm_provider(config: Optional[LLMConfig] = None) -> LLMProvider:
    """Factory : instancie le bon fournisseur selon la configuration.

    C'est le SEUL endroit du code qui a besoin de connaître les différents
    fournisseurs. Tout le reste de l'application (RAG, scoring, API) manipule
    uniquement l'interface abstraite LLMProvider.
    """
    cfg = config or llm_config
    provider_map = {
        "mock": lambda: MockLLMProvider(),
        "openai": lambda: OpenAIProvider(cfg),
        "azure_openai": lambda: AzureOpenAIProvider(cfg),
        "ollama": lambda: OllamaProvider(cfg),
    }
    factory = provider_map.get(cfg.provider)
    if factory is None:
        raise ValueError(
            f"Fournisseur LLM inconnu : '{cfg.provider}'. "
            f"Valeurs possibles : {list(provider_map.keys())}"
        )
    provider = factory()
    logger.info("Fournisseur LLM initialisé : %s", provider.name)
    return provider
