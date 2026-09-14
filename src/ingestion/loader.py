"""
Chargement de la base de connaissances (référentiels de sécurité au format JSON)
et transformation en "documents" prêts à être indexés par le RAG.

Point clé pour la traçabilité (exigée dans un contexte de conformité) :
chaque chunk indexé conserve ses métadonnées d'origine (id de contrôle,
référentiel, domaine) afin que le chatbot puisse toujours CITER sa source
exacte, et non produire une réponse non vérifiable.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from src.config import KNOWLEDGE_BASE_DIR

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeChunk:
    """Unité indexable : un contrôle/exigence de référentiel avec ses métadonnées."""

    chunk_id: str
    text: str
    metadata: Dict = field(default_factory=dict)


def _control_to_text(control: dict) -> str:
    """Transforme un contrôle JSON en texte dense, optimisé pour la recherche
    sémantique (titre + résumé + questions d'évaluation concaténés)."""
    questions = "\n".join(f"- {q}" for q in control.get("questions_evaluation", []))
    return (
        f"[{control['referentiel']}] {control['id']} — {control['titre']} "
        f"(domaine : {control['domaine']})\n"
        f"{control['resume']}\n"
        f"Questions d'évaluation associées :\n{questions}"
    )


def load_knowledge_base(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> List[KnowledgeChunk]:
    """Charge tous les fichiers JSON du dossier data/knowledge_base/ et les
    convertit en chunks indexables.

    Pour ajouter un nouveau référentiel (ex. DORA, PCI-DSS, un référentiel
    interne d'entreprise), il suffit d'ajouter un fichier JSON respectant le
    même schéma dans ce dossier — aucune modification de code nécessaire.
    """
    chunks: List[KnowledgeChunk] = []
    json_files = sorted(kb_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"Aucun fichier JSON trouvé dans {kb_dir}. "
            "Vérifiez que la base de connaissances a bien été fournie."
        )

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            controls = json.load(f)
        for control in controls:
            chunks.append(
                KnowledgeChunk(
                    chunk_id=control["id"],
                    text=_control_to_text(control),
                    metadata={
                        "referentiel": control["referentiel"],
                        "domaine": control["domaine"],
                        "titre": control["titre"],
                        "niveau_criticite_defaut": control.get("niveau_criticite_defaut", "moyen"),
                        "source_file": file_path.name,
                    },
                )
            )
    logger.info("Base de connaissances chargée : %d contrôles issus de %d fichier(s).", len(chunks), len(json_files))
    return chunks


def load_raw_controls(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> List[dict]:
    """Charge les contrôles bruts (format dict), utilisé par le moteur de
    scoring qui a besoin des questions d'évaluation individuelles, pas
    seulement du texte concaténé."""
    controls: List[dict] = []
    for file_path in sorted(kb_dir.glob("*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            controls.extend(json.load(f))
    return controls
