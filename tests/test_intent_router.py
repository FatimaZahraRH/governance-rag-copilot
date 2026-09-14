"""Tests unitaires du routeur d'intention (partie règles, sans dépendance LLM)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.intent_router import Intent, _rule_based_intent


class TestRuleBasedIntent:
    def test_demande_evaluation_detectee(self):
        result = _rule_based_intent("Peux-tu évaluer ce nouveau fournisseur SaaS pour la paie ?")
        assert result is not None
        assert result.intent == Intent.DEMANDE_EVALUATION

    def test_escalade_humaine_detectee_sur_urgence(self):
        result = _rule_based_intent("Nous avons une fuite de données en cours, c'est urgent !")
        assert result is not None
        assert result.intent == Intent.ESCALADE_HUMAINE

    def test_escalade_prioritaire_sur_evaluation_si_les_deux_presents(self):
        """Si un message contient à la fois des mots-clés d'urgence et
        d'évaluation, l'escalade humaine doit primer (principe de précaution :
        mieux vaut sur-escalader que sous-réagir à un incident)."""
        result = _rule_based_intent("Incident en cours sur le fournisseur qu'on vient d'évaluer, c'est urgent")
        assert result is not None
        assert result.intent == Intent.ESCALADE_HUMAINE

    def test_question_generale_ne_matche_aucune_regle(self):
        """Une question factuelle simple ne doit déclencher aucune règle —
        elle sera alors traitée par le RAG classique en amont du routeur."""
        result = _rule_based_intent("Qu'est-ce que le chiffrement AES ?")
        assert result is None
