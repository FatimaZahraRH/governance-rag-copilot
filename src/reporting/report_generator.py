"""
Génère un rapport d'évaluation lisible (Markdown) à partir d'un EvaluationReport.

Objectif : ne pas se contenter de retourner un score brut (peu actionnable),
mais produire un vrai livrable avec un plan de remédiation priorisé — ce
qu'attend une équipe Security by Design d'un outil d'évaluation de tiers.
"""
from __future__ import annotations

from datetime import datetime

from src.scoring.engine import ComplianceScoringEngine
from src.scoring.models import EvaluationReport, NiveauReponse

_RISK_EMOJI = {
    "faible": "🟢",
    "moyen": "🟡",
    "eleve": "🟠",
    "critique": "🔴",
}

_REPONSE_LABEL = {
    NiveauReponse.CONFORME: "Conforme",
    NiveauReponse.PARTIEL: "Partiellement conforme",
    NiveauReponse.NON_CONFORME: "Non conforme",
    NiveauReponse.NON_APPLICABLE: "Non applicable",
}


def generate_markdown_report(report: EvaluationReport, engine: ComplianceScoringEngine) -> str:
    ctx = report.project_context
    emoji = _RISK_EMOJI.get(report.niveau_risque_global, "")
    lines: list[str] = []

    lines.append(f"# Rapport d'évaluation de sécurité — {ctx.nom_projet}")
    lines.append(f"*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*\n")

    lines.append("## Synthèse")
    lines.append(f"- **Score de conformité pondéré : {report.score_global_pourcent} %**")
    lines.append(f"- **Niveau de risque global : {emoji} {report.niveau_risque_global.upper()}**")
    lines.append(f"- Facteur d'amplification contextuel appliqué : ×{ctx.facteur_amplification()}\n")

    lines.append("## Contexte du projet évalué")
    lines.append(f"- Traite des données personnelles : {'Oui' if ctx.traite_donnees_personnelles else 'Non'}")
    lines.append(f"- Traite des données sensibles (santé, finance...) : {'Oui' if ctx.traite_donnees_sensibles else 'Non'}")
    lines.append(f"- Exposé sur Internet : {'Oui' if ctx.expose_sur_internet else 'Non'}")
    lines.append(f"- Fournisseur tiers : {'Oui' if ctx.fournisseur_tiers else 'Non'}")
    lines.append(f"- Volumétrie utilisateurs : {ctx.volumetrie_utilisateurs}\n")

    lines.append("## Plan de remédiation priorisé")
    lines.append(
        "*Les contrôles ci-dessous sont triés par poids effectif décroissant "
        "(criticité du référentiel × sensibilité du contexte projet), pas par ordre alphabétique.*\n"
    )
    prioritized = engine.get_prioritized_remediation(report)
    if not prioritized:
        lines.append("Aucune non-conformité détectée. ✅\n")
    else:
        lines.append("| Priorité | Contrôle | Référentiel | Statut | Poids effectif |")
        lines.append("|---|---|---|---|---|")
        for i, result in enumerate(prioritized, start=1):
            lines.append(
                f"| {i} | {result.control_id} — {result.titre} | {result.referentiel} "
                f"| {_REPONSE_LABEL[result.reponse]} | {result.poids_effectif} |"
            )
        lines.append("")

    lines.append("## Détail de tous les contrôles évalués")
    lines.append("| Contrôle | Domaine | Statut | Score pondéré / Poids max |")
    lines.append("|---|---|---|---|")
    for result in report.control_results:
        lines.append(
            f"| {result.control_id} | {result.domaine} | {_REPONSE_LABEL[result.reponse]} "
            f"| {result.score_pondere} / {result.poids_effectif} |"
        )

    lines.append("\n---")
    lines.append(
        "*Rapport généré automatiquement par le moteur de scoring de conformité. "
        "Toute décision d'homologation reste soumise à validation par l'équipe "
        "Security by Design.*"
    )

    return "\n".join(lines)


def save_report(report: EvaluationReport, engine: ComplianceScoringEngine, output_path: str) -> None:
    markdown = generate_markdown_report(report, engine)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)
