#!/usr/bin/env python3
"""
Évaluation quantifiée de la qualité du RAG — c'est ce qui distingue ce
projet d'un simple "chatbot qui marche à l'œil". On mesure ici :

  - Recall@k    : parmi les contrôles réellement pertinents pour une
                  question, quelle proportion est effectivement retrouvée
                  dans le top-k ?
  - Precision@k : parmi les k contrôles retournés, quelle proportion est
                  réellement pertinente ?
  - Taux de rejet correct (groundedness) : pour les questions hors-sujet
                  (ex. "recette de tarte au citron"), le système doit
                  retourner AUCUNE source plutôt que d'halluciner une
                  réponse sourcée sur du contenu non pertinent.

Usage :
    python evaluation/eval_rag.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.embeddings import get_embedding_provider
from src.rag.retriever import Retriever
from src.rag.vectorstore import VectorStore

logging.basicConfig(level=logging.WARNING)

TEST_SET_PATH = Path(__file__).resolve().parent / "test_questions.json"


def load_test_set() -> list[dict]:
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(retriever: Retriever, test_set: list[dict], top_k: int = 4) -> dict:
    recalls: list[float] = []
    precisions: list[float] = []
    groundedness_errors = 0
    per_question_results = []

    for case in test_set:
        question = case["question"]
        expected = set(case["expected_control_ids"])
        passages = retriever.retrieve(question, top_k=top_k)
        retrieved_ids = {p.chunk_id for p in passages}

        if not expected:
            # Question hors-sujet : le système DOIT ne rien retrouver de pertinent.
            is_correct_rejection = len(retrieved_ids) == 0
            if not is_correct_rejection:
                groundedness_errors += 1
            per_question_results.append({
                "question": question,
                "type": "hors_sujet",
                "rejet_correct": is_correct_rejection,
                "retrieved": sorted(retrieved_ids),
            })
            continue

        true_positives = expected & retrieved_ids
        recall = len(true_positives) / len(expected) if expected else 1.0
        precision = len(true_positives) / len(retrieved_ids) if retrieved_ids else 0.0

        recalls.append(recall)
        precisions.append(precision)
        per_question_results.append({
            "question": question,
            "type": "factuelle",
            "expected": sorted(expected),
            "retrieved": sorted(retrieved_ids),
            "recall": round(recall, 2),
            "precision": round(precision, 2),
        })

    return {
        "mean_recall_at_k": round(mean(recalls), 3) if recalls else None,
        "mean_precision_at_k": round(mean(precisions), 3) if precisions else None,
        "groundedness_errors": groundedness_errors,
        "n_questions_factuelles": len(recalls),
        "n_questions_hors_sujet": len(test_set) - len(recalls),
        "details": per_question_results,
    }


def main() -> None:
    print("Chargement du vector store et du retriever...")
    embedding_provider = get_embedding_provider()
    store = VectorStore(embedding_provider)
    if store.count() == 0:
        print("⚠️  Vector store vide. Lancez d'abord : python scripts/build_index.py")
        return
    retriever = Retriever(store)

    test_set = load_test_set()
    print(f"Évaluation sur {len(test_set)} questions de test...\n")

    results = evaluate_retrieval(retriever, test_set, top_k=4)

    print("=" * 60)
    print("RÉSULTATS DE L'ÉVALUATION DU RETRIEVAL")
    print("=" * 60)
    print(f"Recall@4 moyen (questions factuelles)    : {results['mean_recall_at_k']}")
    print(f"Precision@4 moyenne (questions factuelles): {results['mean_precision_at_k']}")
    print(f"Erreurs de groundedness (hors-sujet mal traité) : "
          f"{results['groundedness_errors']} / {results['n_questions_hors_sujet']}")
    print()
    for detail in results["details"]:
        if detail["type"] == "factuelle":
            print(f"  [{detail['recall']:.2f} recall | {detail['precision']:.2f} precision] {detail['question']}")
        else:
            status = "✅ rejet correct" if detail["rejet_correct"] else "❌ hallucination de source"
            print(f"  [{status}] {detail['question']}")

    output_path = Path(__file__).resolve().parent.parent / "evaluation_rag_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Résultats détaillés sauvegardés dans : {output_path}")


if __name__ == "__main__":
    main()
