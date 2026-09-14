#!/usr/bin/env python3
"""
Script d'indexation : charge la base de connaissances (référentiels JSON) et
construit/rafraîchit le vector store local.

Usage :
    python scripts/build_index.py
    python scripts/build_index.py --reset   # ré-indexe tout depuis zéro
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.loader import load_knowledge_base
from src.rag.embeddings import get_embedding_provider
from src.rag.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexe la base de connaissances dans le vector store.")
    parser.add_argument("--reset", action="store_true", help="Supprime et recrée la collection avant indexation.")
    args = parser.parse_args()

    logger.info("Chargement de la base de connaissances...")
    chunks = load_knowledge_base()

    logger.info("Initialisation du fournisseur d'embeddings...")
    embedding_provider = get_embedding_provider()

    logger.info("Connexion au vector store...")
    store = VectorStore(embedding_provider)

    if args.reset:
        logger.info("Réinitialisation de la collection demandée (--reset).")
        store.reset()

    logger.info("Indexation de %d chunks...", len(chunks))
    store.add_chunks(chunks)

    logger.info("Terminé. Nombre total de documents indexés : %d", store.count())


if __name__ == "__main__":
    main()
