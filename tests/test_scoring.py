"""
Tests unitaires du moteur de scoring de conformité.

Lancement : pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.scoring.criticality import (
    classify_global_risk,
    compute_effective_weight,
    compute_weighted_score,
)
from src.scoring.engine import ComplianceScoringEngine
from src.scoring.models import (
    ControlAnswer,
    NiveauCriticite,
    NiveauReponse,
    ProjectContext,
)


class TestProjectContextAmplification:
    def test_projet_neutre_facteur_unitaire(self):
        """Un projet sans aucun facteur de risque garde un facteur de 1.0."""
        ctx = ProjectContext(nom_projet="Outil interne basique")
        assert ctx.facteur_amplification() == 1.0

    def test_donnees_personnelles_augmente_le_facteur(self):
        ctx = ProjectContext(nom_projet="X", traite_donnees_personnelles=True)
        assert ctx.facteur_amplification() == 1.3

    def test_cumul_de_facteurs_de_risque(self):
        """Plusieurs facteurs de risque se cumulent, ce qui reflète bien
        qu'un projet exposé + données sensibles + fort trafic est
        significativement plus critique qu'un projet neutre."""
        ctx = ProjectContext(
            nom_projet="SaaS RH critique",
            traite_donnees_personnelles=True,
            traite_donnees_sensibles=True,
            expose_sur_internet=True,
            fournisseur_tiers=True,
            volumetrie_utilisateurs="elevee",
        )
        # 1.0 + 0.3 + 0.5 + 0.3 + 0.2 + 0.3 = 2.6
        assert ctx.facteur_amplification() == 2.6


class TestCriticalityWeighting:
    def test_poids_effectif_augmente_avec_amplification(self):
        ctx_neutre = ProjectContext(nom_projet="A")
        ctx_critique = ProjectContext(nom_projet="B", traite_donnees_sensibles=True, expose_sur_internet=True)

        poids_neutre = compute_effective_weight(NiveauCriticite.ELEVE, ctx_neutre)
        poids_critique = compute_effective_weight(NiveauCriticite.ELEVE, ctx_critique)

        assert poids_critique > poids_neutre

    def test_score_pondere_conforme_egal_au_poids(self):
        poids = 3.0
        score = compute_weighted_score(NiveauReponse.CONFORME, poids)
        assert score == poids

    def test_score_pondere_non_conforme_est_nul(self):
        score = compute_weighted_score(NiveauReponse.NON_CONFORME, 3.0)
        assert score == 0.0

    def test_score_pondere_partiel_est_la_moitie(self):
        score = compute_weighted_score(NiveauReponse.PARTIEL, 4.0)
        assert score == 2.0


class TestGlobalRiskClassification:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (95.0, "faible"),
            (85.0, "faible"),
            (84.9, "moyen"),
            (65.0, "moyen"),
            (64.9, "eleve"),
            (40.0, "eleve"),
            (39.9, "critique"),
            (0.0, "critique"),
        ],
    )
    def test_seuils_de_classification(self, score, expected):
        assert classify_global_risk(score) == expected


class TestComplianceScoringEngine:
    @pytest.fixture
    def engine(self):
        return ComplianceScoringEngine()

    def test_liste_controles_non_vide(self, engine):
        controls = engine.list_available_controls()
        assert len(controls) > 0
        assert all("id" in c for c in controls)

    def test_evaluation_tous_conformes_donne_score_maximal(self, engine):
        ctx = ProjectContext(nom_projet="Projet exemplaire")
        all_ids = [c["id"] for c in engine.list_available_controls()]
        answers = [ControlAnswer(control_id=cid, reponse=NiveauReponse.CONFORME) for cid in all_ids]

        report = engine.evaluate(ctx, answers)

        assert report.score_global_pourcent == 100.0
        assert report.niveau_risque_global == "faible"

    def test_evaluation_tous_non_conformes_donne_score_nul(self, engine):
        ctx = ProjectContext(nom_projet="Projet à risque")
        all_ids = [c["id"] for c in engine.list_available_controls()]
        answers = [ControlAnswer(control_id=cid, reponse=NiveauReponse.NON_CONFORME) for cid in all_ids]

        report = engine.evaluate(ctx, answers)

        assert report.score_global_pourcent == 0.0
        assert report.niveau_risque_global == "critique"

    def test_controle_inconnu_est_ignore_sans_erreur(self, engine):
        ctx = ProjectContext(nom_projet="Projet test")
        answers = [ControlAnswer(control_id="INEXISTANT-999", reponse=NiveauReponse.CONFORME)]

        report = engine.evaluate(ctx, answers)

        assert report.control_results == []
        assert report.score_global_pourcent == 0.0

    def test_remediation_priorisee_triee_par_poids_decroissant(self, engine):
        ctx = ProjectContext(nom_projet="Projet mixte", traite_donnees_sensibles=True, expose_sur_internet=True)
        answers = [
            ControlAnswer("ISO27001-A.5.1", NiveauReponse.NON_CONFORME),   # criticité moyenne
            ControlAnswer("ISO27001-A.8.24", NiveauReponse.NON_CONFORME),  # criticité élevée (crypto)
        ]
        report = engine.evaluate(ctx, answers)

        prioritized = engine.get_prioritized_remediation(report)

        assert len(prioritized) == 2
        # Le contrôle de criticité élevée doit apparaître en premier
        assert prioritized[0].control_id == "ISO27001-A.8.24"
        assert prioritized[0].poids_effectif >= prioritized[1].poids_effectif

    def test_non_applicable_exclu_du_calcul(self, engine):
        ctx = ProjectContext(nom_projet="Projet test")
        answers = [
            ControlAnswer("ISO27001-A.5.1", NiveauReponse.CONFORME),
            ControlAnswer("ISO27001-A.5.23", NiveauReponse.NON_APPLICABLE),
        ]
        report = engine.evaluate(ctx, answers)
        # Seul le contrôle applicable doit compter dans le score -> 100%
        assert report.score_global_pourcent == 100.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
