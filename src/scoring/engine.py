"""
Moteur d'évaluation de conformité / sécurité des tiers (ISP).

Reproduit la mission n°1 de l'offre VINCI Construction SI : "contribuer à
l'intégration de la cybersécurité dans les projets internes et aux
évaluations de sécurité des tiers, incluant les exigences réglementaires et
contractuelles".

Usage typique :
    engine = ComplianceScoringEngine()
    context = ProjectContext(nom_projet="Outil RH SaaS", traite_donnees_personnelles=True, ...)
    answers = [ControlAnswer(control_id="ISO27001-A.8.24", reponse=NiveauReponse.PARTIEL), ...]
    report = engine.evaluate(context, answers)
"""
from __future__ import annotations

import logging
from typing import List

from src.ingestion.loader import load_raw_controls
from src.scoring.criticality import (
    classify_global_risk,
    compute_effective_weight,
    compute_weighted_score,
)
from src.scoring.models import (
    ControlAnswer,
    ControlResult,
    EvaluationReport,
    NiveauCriticite,
    NiveauReponse,
    ProjectContext,
)

logger = logging.getLogger(__name__)


class ComplianceScoringEngine:
    def __init__(self):
        self._controls_by_id = {c["id"]: c for c in load_raw_controls()}

    def list_available_controls(self) -> List[dict]:
        """Utile pour générer dynamiquement un questionnaire d'évaluation
        (front-end, API) à partir de la base de connaissances."""
        return list(self._controls_by_id.values())

    def evaluate(self, project_context: ProjectContext, answers: List[ControlAnswer]) -> EvaluationReport:
        results: List[ControlResult] = []
        total_weight = 0.0
        total_score = 0.0

        for answer in answers:
            control = self._controls_by_id.get(answer.control_id)
            if control is None:
                logger.warning("Contrôle inconnu ignoré : %s", answer.control_id)
                continue

            niveau_criticite = NiveauCriticite(control.get("niveau_criticite_defaut", "moyen"))
            poids_effectif = compute_effective_weight(niveau_criticite, project_context)
            score_pondere = compute_weighted_score(answer.reponse, poids_effectif)

            if answer.reponse != NiveauReponse.NON_APPLICABLE:
                total_weight += poids_effectif
                total_score += score_pondere

            results.append(
                ControlResult(
                    control_id=control["id"],
                    titre=control["titre"],
                    referentiel=control["referentiel"],
                    domaine=control["domaine"],
                    reponse=answer.reponse,
                    niveau_criticite=niveau_criticite,
                    poids_effectif=poids_effectif,
                    score_pondere=score_pondere,
                )
            )

        score_global_pourcent = round((total_score / total_weight) * 100, 1) if total_weight > 0 else 0.0
        niveau_risque_global = classify_global_risk(score_global_pourcent)

        report = EvaluationReport(
            project_context=project_context,
            control_results=results,
            score_global_pourcent=score_global_pourcent,
            niveau_risque_global=niveau_risque_global,
        )
        logger.info(
            "Évaluation '%s' terminée : score=%.1f%%, risque=%s",
            project_context.nom_projet, score_global_pourcent, niveau_risque_global,
        )
        return report

    def get_prioritized_remediation(self, report: EvaluationReport) -> List[ControlResult]:
        """Trie les contrôles non conformes/partiels par poids effectif
        décroissant : ce qui doit être corrigé EN PREMIER, ce n'est pas
        forcément ce qui a le score le plus bas, mais ce qui pèse le plus
        lourd dans le contexte du projet évalué."""
        non_conformes = [
            r for r in report.control_results
            if r.reponse in (NiveauReponse.NON_CONFORME, NiveauReponse.PARTIEL)
        ]
        return sorted(non_conformes, key=lambda r: r.poids_effectif, reverse=True)
