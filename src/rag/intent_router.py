"""
Routeur d'intention.

L'offre de stage VINCI Construction SI (Security by Design) décrit
explicitement le besoin : "répondre aux questions récurrentes des chefs de
projet ET les orienter vers les bons référentiels, interlocuteurs ou
livrables". Un chatbot RAG "brut" ne fait que répondre à des questions — il
ne sait pas qu'une demande dépasse son périmètre et doit être escaladée.

Ce module classe chaque message entrant en 3 catégories avant même
d'interroger le RAG :

  1. QUESTION_GENERALE   : question factuelle sur un référentiel -> RAG direct
  2. DEMANDE_EVALUATION  : demande d'évaluation d'un projet/fournisseur
                           concret -> doit être redirigée vers le moteur de
                           scoring (src/scoring), pas vers une simple réponse RAG
  3. ESCALADE_HUMAINE    : la question sort du périmètre couvert par la base
                           de connaissances, ou nécessite une décision humaine
                           (dérogation, cas juridique ambigu, urgence) -> le
                           chatbot doit orienter vers un interlocuteur, pas
                           tenter de répondre à sa place

Implémentation volontairement hybride :
  - des règles simples (mots-clés) pour les cas évidents et bon marché en
    calcul, typiques d'un premier niveau de filtrage en production,
  - un classifieur LLM en repli pour les cas ambigus, avec justification
    pour rester auditable.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum

from src.rag.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    QUESTION_GENERALE = "question_generale"
    DEMANDE_EVALUATION = "demande_evaluation"
    ESCALADE_HUMAINE = "escalade_humaine"


@dataclass
class IntentDecision:
    intent: Intent
    confidence: float
    justification: str
    method: str  # "regles" ou "llm"


_EVALUATION_KEYWORDS = [
    "évaluer", "evaluer", "évaluation", "evaluation", "auditer", "audit",
    "nouveau fournisseur", "nouveau prestataire", "nouvel outil", "saas",
    "faire homologuer", "homologation", "revue de sécurité",
    "peux-tu analyser", "peux tu analyser", "analyser ce projet",
]

_ESCALADE_KEYWORDS = [
    "urgent", "incident en cours", "fuite de données", "violation de données",
    "dérogation", "derogation", "je ne suis pas d'accord", "litige", "juridique",
    "avocat", "contentieux", "qui dois-je contacter", "qui contacter",
]


def _rule_based_intent(message: str) -> IntentDecision | None:
    lowered = message.lower()

    if any(kw in lowered for kw in _ESCALADE_KEYWORDS):
        return IntentDecision(
            intent=Intent.ESCALADE_HUMAINE,
            confidence=0.9,
            justification="Mots-clés d'urgence/juridique/incident détectés — orientation vers un interlocuteur humain requise.",
            method="regles",
        )

    if any(kw in lowered for kw in _EVALUATION_KEYWORDS):
        return IntentDecision(
            intent=Intent.DEMANDE_EVALUATION,
            confidence=0.85,
            justification="Demande explicite d'évaluation/audit d'un projet ou fournisseur — redirection vers le moteur de scoring.",
            method="regles",
        )

    return None


_CLASSIFIER_SYSTEM_PROMPT = """Tu es un classifieur d'intention pour un chatbot interne \
de gouvernance cybersécurité (équipe Security by Design). Classe le message de l'utilisateur \
dans une des 3 catégories suivantes, et réponds UNIQUEMENT au format JSON strict :

{"intent": "question_generale" | "demande_evaluation" | "escalade_humaine", \
"confidence": <float entre 0 et 1>, "justification": "<une phrase>"}

Règles :
- "question_generale" : question factuelle sur un référentiel, une bonne pratique, une définition.
- "demande_evaluation" : l'utilisateur veut faire évaluer/auditer un projet, un fournisseur, \
un outil SaaS concret.
- "escalade_humaine" : incident de sécurité en cours, désaccord, cas juridique ambigu, \
dérogation, ou tout sujet hors du périmètre d'un chatbot informatif.
"""


def classify_intent(message: str, llm: LLMProvider) -> IntentDecision:
    """Classifie l'intention d'un message. Essaie d'abord les règles rapides
    (peu coûteuses), puis retombe sur le LLM pour les cas ambigus."""
    rule_result = _rule_based_intent(message)
    if rule_result is not None:
        return rule_result

    raw_response = llm.generate(system_prompt=_CLASSIFIER_SYSTEM_PROMPT, user_prompt=message, temperature=0.0)

    # Le MockLLMProvider ne renvoie pas de JSON valide : on prévoit un repli
    # explicite pour que la démo reste fonctionnelle sans clé API.
    try:
        parsed = json.loads(raw_response)
        return IntentDecision(
            intent=Intent(parsed["intent"]),
            confidence=float(parsed.get("confidence", 0.6)),
            justification=parsed.get("justification", ""),
            method="llm",
        )
    except Exception:
        logger.info("Classification LLM non parsable (attendu en mode mock) — repli sur question_generale.")
        return IntentDecision(
            intent=Intent.QUESTION_GENERALE,
            confidence=0.5,
            justification="Repli par défaut (mode mock ou réponse LLM non structurée).",
            method="fallback",
        )
