"""
Chatbot de gouvernance cyber — orchestrateur principal.

Pipeline complet pour chaque message utilisateur :
  1. Classification d'intention (question / évaluation / escalade)
  2. Si "demande_evaluation"  -> réponse d'orientation vers le moteur de scoring
  3. Si "escalade_humaine"    -> réponse d'orientation vers un interlocuteur, PAS de tentative de réponse
  4. Si "question_generale"   -> RAG classique : retrieve + generate, avec
                                   citation systématique des sources utilisées

Exigence explicite de l'offre VINCI Construction SI reproduite ici : "capacité
d'analyse, de synthèse et de vulgarisation (savoir rendre accessibles des
exigences cyber à des chefs de projet non-experts)". Le prompt système est
donc conçu pour imposer un langage simple, sans jargon non expliqué.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from src.rag.intent_router import Intent, IntentDecision, classify_intent
from src.rag.llm_provider import LLMProvider
from src.rag.retriever import RetrievedPassage, Retriever

logger = logging.getLogger(__name__)


@dataclass
class ChatbotResponse:
    answer: str
    intent: IntentDecision
    sources: List[RetrievedPassage] = field(default_factory=list)
    grounded: bool = True  # False si aucune source pertinente trouvée (question_generale uniquement)


_SYSTEM_PROMPT = """Tu es l'assistant interne de l'équipe Security by Design (SbD). \
Ton rôle est d'aider des chefs de projet, qui ne sont PAS des experts en cybersécurité, \
à comprendre les exigences de sécurité et de conformité qui s'appliquent à leurs projets.

Règles impératives :
1. Réponds UNIQUEMENT à partir des extraits de référentiels fournis dans le contexte ci-dessous. \
Si le contexte ne permet pas de répondre, dis-le explicitement — n'invente jamais une exigence.
2. Vulgarise systématiquement : évite le jargon non expliqué, utilise des phrases courtes, \
donne un exemple concret quand c'est utile.
3. Cite toujours l'identifiant du contrôle utilisé (ex. "ISO27001-A.5.15") pour que la réponse \
soit vérifiable.
4. Reste concis : 4 à 6 phrases maximum, sauf si la question demande explicitement plus de détail.
"""


class GovernanceChatbot:
    def __init__(self, retriever: Retriever, llm: LLMProvider):
        self._retriever = retriever
        self._llm = llm

    def ask(self, message: str) -> ChatbotResponse:
        intent_decision = classify_intent(message, self._llm)
        logger.info(
            "Intention détectée : %s (confiance=%.2f, méthode=%s)",
            intent_decision.intent, intent_decision.confidence, intent_decision.method,
        )

        if intent_decision.intent == Intent.DEMANDE_EVALUATION:
            return self._respond_evaluation_redirect(intent_decision)

        if intent_decision.intent == Intent.ESCALADE_HUMAINE:
            return self._respond_human_escalation(intent_decision)

        return self._respond_general_question(message, intent_decision)

    def _respond_general_question(self, message: str, intent_decision: IntentDecision) -> ChatbotResponse:
        passages = self._retriever.retrieve(message)

        if not passages:
            return ChatbotResponse(
                answer=(
                    "Je n'ai pas trouvé d'exigence pertinente dans la base de connaissances "
                    "actuelle (ISO 27001, NIS2, RGPD) pour répondre précisément à cette question. "
                    "Je te recommande de contacter l'équipe Security by Design directement, ou de "
                    "reformuler ta question avec plus de contexte."
                ),
                intent=intent_decision,
                sources=[],
                grounded=False,
            )

        context = "\n\n".join(
            f"[Source {p.chunk_id}] {p.text}" for p in passages
        )
        user_prompt = (
            f"Question du chef de projet : {message}\n\n"
            f"Contexte (extraits de référentiels pertinents) :\n{context}\n\n"
            "Réponds en respectant les règles du prompt système, et termine ta réponse par "
            "la liste des identifiants de contrôles utilisés entre crochets."
        )
        answer = self._llm.generate(system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt)

        return ChatbotResponse(answer=answer, intent=intent_decision, sources=passages, grounded=True)

    def _respond_evaluation_redirect(self, intent_decision: IntentDecision) -> ChatbotResponse:
        answer = (
            "Ta demande concerne l'évaluation de sécurité d'un projet ou d'un fournisseur. "
            "Ce type de demande nécessite une évaluation structurée plutôt qu'une réponse "
            "rapide de ma part : je te redirige vers le module d'évaluation de conformité "
            "(voir src/scoring/engine.py), qui va te poser une série de questions ciblées "
            "et générer un rapport de risque avec un plan de remédiation priorisé."
        )
        return ChatbotResponse(answer=answer, intent=intent_decision, sources=[], grounded=True)

    def _respond_human_escalation(self, intent_decision: IntentDecision) -> ChatbotResponse:
        answer = (
            "Ta demande semble nécessiter l'intervention d'un membre de l'équipe Security by "
            "Design (urgence, désaccord, ou cas hors du périmètre couvert par mes réponses). "
            "Je t'invite à contacter directement l'équipe SbD plutôt que d'attendre une réponse "
            "automatisée sur ce sujet."
        )
        return ChatbotResponse(answer=answer, intent=intent_decision, sources=[], grounded=True)
