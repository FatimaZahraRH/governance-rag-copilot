"""Modèles de données du moteur de scoring de conformité."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class NiveauReponse(str, Enum):
    """Niveau de mise en œuvre d'un contrôle, déclaré par l'évalué (fournisseur,
    chef de projet) ou constaté par l'évaluateur."""
    CONFORME = "conforme"           # 1.0
    PARTIEL = "partiel"             # 0.5
    NON_CONFORME = "non_conforme"   # 0.0
    NON_APPLICABLE = "non_applicable"  # exclu du calcul


class NiveauCriticite(str, Enum):
    """Niveau de criticité du contexte évalué. Détermine la pondération
    appliquée à chaque contrôle (ex. un contrôle "élevé" pèse plus lourd
    si le projet traite des données personnelles sensibles)."""
    FAIBLE = "faible"
    MOYEN = "moyen"
    ELEVE = "eleve"


# Poids appliqué selon le niveau de criticité par défaut du contrôle
_CRITICITE_WEIGHTS = {
    NiveauCriticite.FAIBLE: 1.0,
    NiveauCriticite.MOYEN: 2.0,
    NiveauCriticite.ELEVE: 3.0,
}

_REPONSE_SCORES = {
    NiveauReponse.CONFORME: 1.0,
    NiveauReponse.PARTIEL: 0.5,
    NiveauReponse.NON_CONFORME: 0.0,
}


def criticite_weight(niveau: NiveauCriticite) -> float:
    return _CRITICITE_WEIGHTS[niveau]


def reponse_score(niveau: NiveauReponse) -> float:
    return _REPONSE_SCORES[niveau]


@dataclass
class ProjectContext:
    """Décrit le contexte du projet/fournisseur évalué, utilisé pour moduler
    la criticité de chaque contrôle (un même contrôle ISO 27001 n'a pas le
    même poids selon qu'on évalue un outil interne sans donnée sensible ou
    une solution SaaS traitant des données personnelles de milliers de clients)."""

    nom_projet: str
    traite_donnees_personnelles: bool = False
    traite_donnees_sensibles: bool = False  # santé, données financières, etc.
    expose_sur_internet: bool = False
    fournisseur_tiers: bool = False
    volumetrie_utilisateurs: str = "faible"  # "faible" | "moyenne" | "elevee"

    def facteur_amplification(self) -> float:
        """Calcule un facteur multiplicatif de criticité globale du projet,
        appliqué en plus de la criticité par contrôle. Logique volontairement
        simple et transparente (chaque facteur est explicite et justifiable
        en entretien / en audit), plutôt qu'un modèle boîte noire."""
        facteur = 1.0
        if self.traite_donnees_personnelles:
            facteur += 0.3
        if self.traite_donnees_sensibles:
            facteur += 0.5
        if self.expose_sur_internet:
            facteur += 0.3
        if self.fournisseur_tiers:
            facteur += 0.2
        if self.volumetrie_utilisateurs == "elevee":
            facteur += 0.3
        elif self.volumetrie_utilisateurs == "moyenne":
            facteur += 0.1
        return round(facteur, 2)


@dataclass
class ControlAnswer:
    control_id: str
    reponse: NiveauReponse
    commentaire: str = ""


@dataclass
class ControlResult:
    control_id: str
    titre: str
    referentiel: str
    domaine: str
    reponse: NiveauReponse
    niveau_criticite: NiveauCriticite
    poids_effectif: float
    score_pondere: float  # entre 0 et poids_effectif


@dataclass
class EvaluationReport:
    project_context: ProjectContext
    control_results: List[ControlResult] = field(default_factory=list)
    score_global_pourcent: float = 0.0
    niveau_risque_global: str = ""  # "faible" | "moyen" | "eleve" | "critique"
