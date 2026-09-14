#!/usr/bin/env python3
"""
Script de démonstration end-to-end : montre les 2 briques du projet
(chatbot RAG + moteur de scoring) fonctionner ensemble, sans nécessiter de
clé API (mode LLM_PROVIDER=mock par défaut).

Usage :
    python scripts/demo.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import llm_config
from src.rag.chatbot import GovernanceChatbot
from src.rag.embeddings import get_embedding_provider
from src.rag.llm_provider import get_llm_provider
from src.rag.retriever import Retriever
from src.rag.vectorstore import VectorStore
from src.reporting.report_generator import generate_markdown_report
from src.scoring.engine import ComplianceScoringEngine
from src.scoring.models import ControlAnswer, NiveauReponse, ProjectContext

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")


def separator(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def demo_chatbot() -> None:
    separator("PARTIE 1 — Chatbot de gouvernance cyber (RAG sourcé)")

    embedding_provider = get_embedding_provider()
    store = VectorStore(embedding_provider)
    if store.count() == 0:
        print("⚠️  Le vector store est vide. Lancez d'abord : python scripts/build_index.py")
        return

    retriever = Retriever(store)
    llm = get_llm_provider()
    print(f"Fournisseur LLM actif : {llm.name} (configurable via LLM_PROVIDER dans .env)\n")

    chatbot = GovernanceChatbot(retriever, llm)

    questions = [
        "Quelles sont les exigences de chiffrement pour les données sensibles ?",
        "On veut adopter un nouvel outil SaaS pour la gestion RH, peux-tu évaluer ce fournisseur ?",
        "On a une fuite de données en cours, qui dois-je contacter en urgence ?",
    ]

    for question in questions:
        print(f"\n💬 Question : {question}")
        response = chatbot.ask(question)
        print(f"🔎 Intention détectée : {response.intent.intent.value} "
              f"(confiance={response.intent.confidence:.2f}, méthode={response.intent.method})")
        print(f"🤖 Réponse :\n{response.answer}")
        if response.sources:
            print(f"📚 Sources citées : {[s.as_citation() for s in response.sources]}")


def demo_scoring() -> None:
    separator("PARTIE 2 — Moteur de scoring de conformité (évaluation de tiers)")

    engine = ComplianceScoringEngine()

    # Contexte réaliste : un outil SaaS RH, exposé sur Internet, avec données
    # personnelles -> criticité contextuelle élevée, cf. facteur_amplification()
    context = ProjectContext(
        nom_projet="Outil SaaS de gestion RH (fournisseur tiers)",
        traite_donnees_personnelles=True,
        traite_donnees_sensibles=False,
        expose_sur_internet=True,
        fournisseur_tiers=True,
        volumetrie_utilisateurs="elevee",
    )
    print(f"Contexte évalué : {context.nom_projet}")
    print(f"Facteur d'amplification de criticité calculé : ×{context.facteur_amplification()}\n")

    # Réponses volontairement mixtes pour illustrer un cas réaliste
    answers = [
        ControlAnswer("ISO27001-A.5.1", NiveauReponse.CONFORME),
        ControlAnswer("ISO27001-A.5.9", NiveauReponse.PARTIEL),
        ControlAnswer("ISO27001-A.5.15", NiveauReponse.PARTIEL),
        ControlAnswer("ISO27001-A.5.23", NiveauReponse.CONFORME),
        ControlAnswer("ISO27001-A.8.24", NiveauReponse.NON_CONFORME),  # chiffrement -> critique
        ControlAnswer("RGPD-ART32-SECURITE", NiveauReponse.NON_CONFORME),
        ControlAnswer("RGPD-ART33-VIOLATION", NiveauReponse.CONFORME),
        ControlAnswer("NIS2-ART21-SUPPLYCHAIN", NiveauReponse.PARTIEL),
    ]

    report = engine.evaluate(context, answers)
    print(f"Score de conformité pondéré : {report.score_global_pourcent} %")
    print(f"Niveau de risque global     : {report.niveau_risque_global.upper()}\n")

    markdown = generate_markdown_report(report, engine)
    output_path = Path(__file__).resolve().parent.parent / "evaluation_demo_report.md"
    output_path.write_text(markdown, encoding="utf-8")
    print(f"📄 Rapport complet généré : {output_path}")
    print("\n--- Aperçu du plan de remédiation priorisé ---")
    for i, result in enumerate(engine.get_prioritized_remediation(report), start=1):
        print(f"  {i}. [{result.poids_effectif}] {result.control_id} — {result.titre} ({result.reponse.value})")


if __name__ == "__main__":
    print(f"Configuration LLM active : provider='{llm_config.provider}' "
          f"(changez LLM_PROVIDER dans .env pour tester avec un vrai LLM)")
    demo_chatbot()
    demo_scoring()
    print("\n✅ Démonstration terminée.")
