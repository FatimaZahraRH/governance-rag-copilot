"""
Moteur de pondération contextuelle de la criticité.

C'est la brique qui différencie ce moteur d'un simple "questionnaire coché" :
le même contrôle (ex. chiffrement des données) n'a pas le même poids selon le
contexte du projet évalué. Un défaut de chiffrement sur un outil interne sans
donnée sensible est une non-conformité mineure ; le même défaut sur une
solution SaaS exposée sur Internet et traitant des données de santé est
critique.

La logique reprend directement l'approche de pondération contextuelle déjà
mise en œuvre dans mon projet Cassiopée (GraphRAG de conformité fournisseurs
sur ISO 27001/NIS2/SOC2/GDPR), appliquée ici à un référentiel plus large et à
une architecture entièrement indépendante d'un fournisseur cloud particulier.
"""
from __future__ import annotations

from src.scoring.models import (
    NiveauCriticite,
    NiveauReponse,
    ProjectContext,
    criticite_weight,
    reponse_score,
)


def compute_effective_weight(niveau_defaut: NiveauCriticite, project_context: ProjectContext) -> float:
    """Combine le poids par défaut d'un contrôle (défini dans le référentiel)
    avec le facteur d'amplification lié au contexte du projet évalué."""
    base_weight = criticite_weight(niveau_defaut)
    amplification = project_context.facteur_amplification()
    return round(base_weight * amplification, 3)


def compute_weighted_score(reponse: NiveauReponse, poids_effectif: float) -> float:
    """Score pondéré d'un contrôle : poids_effectif * score de la réponse
    (0 si non conforme, 0.5 si partiel, 1 si conforme)."""
    if reponse == NiveauReponse.NON_APPLICABLE:
        return 0.0
    return round(poids_effectif * reponse_score(reponse), 3)


def classify_global_risk(score_pourcent: float) -> str:
    """Classification du niveau de risque global à partir du score de
    conformité pondéré, exprimé en pourcentage (0-100).

    Seuils choisis de façon à rester conservateurs : un score de 80% ne
    suffit PAS à qualifier un fournisseur de "faible risque", car les
    contrôles élevés (chiffrement, contrôle d'accès...) pèsent davantage
    dans le calcul pondéré — un score global élevé masquant un contrôle
    critique non conforme reste visible dans le détail du rapport.
    """
    if score_pourcent >= 85:
        return "faible"
    if score_pourcent >= 65:
        return "moyen"
    if score_pourcent >= 40:
        return "eleve"
    return "critique"
