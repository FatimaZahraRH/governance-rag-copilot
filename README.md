# Governance RAG Copilot

**Chatbot de gouvernance cybersécurité (RAG sourcé) + moteur de scoring de conformité pour l'évaluation de sécurité des tiers.**

Projet réalisé pour préparer une candidature en cybersécurité orientée *Security by Design*, gouvernance cyber et IA générative appliquée à la conformité (ISO 27001, NIS2, RGPD).

---

## 🎯 Pourquoi ce projet

Les équipes de gouvernance cybersécurité passent beaucoup de temps à répondre à des questions récurrentes des chefs de projet ("dois-je chiffrer ces données ?", "ce fournisseur SaaS est-il conforme ?") et à évaluer la sécurité de solutions tierces avant leur adoption. Ce projet construit les deux briques correspondantes :

1. **Un chatbot RAG** qui répond à des questions de gouvernance en citant systématiquement sa source (aucune réponse non vérifiable), vulgarise pour un public non expert, et **route intelligemment** les demandes qui ne relèvent pas d'une simple question (évaluation à mener, incident à escalader).
2. **Un moteur de scoring de conformité** qui évalue un projet ou un fournisseur tiers sur une liste de contrôles, en pondérant chaque contrôle selon la criticité réelle du contexte (un défaut de chiffrement pèse plus lourd sur un SaaS exposé traitant des données de santé que sur un outil interne sans donnée sensible), et génère un rapport avec plan de remédiation priorisé.

---

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│  Question   │─────▶│  Routeur         │─────▶│ question_generale    │──▶ RAG (retrieve + generate, sourcé)
│  utilisateur│      │  d'intention     │      │ demande_evaluation   │──▶ Redirection moteur de scoring
└─────────────┘      └──────────────────┘      │ escalade_humaine     │──▶ Redirection interlocuteur humain
                                                 └─────────────────────┘

┌──────────────────┐    ┌───────────────┐    ┌──────────────────┐    ┌────────────────────┐
│ Base de          │───▶│ Embeddings    │───▶│ Vector store      │───▶│ Retriever            │
│ connaissances     │    │ (agnostique)  │    │ (ChromaDB local)  │    │ (seuil de similarité)│
│ ISO27001/NIS2/RGPD│    └───────────────┘    └──────────────────┘    └────────────────────┘

┌──────────────────┐    ┌────────────────────┐    ┌─────────────────────┐
│ Contexte projet  │───▶│ Moteur de scoring  │───▶│ Rapport Markdown      │
│ (criticité)      │    │ (pondération        │    │ + plan de remédiation│
│ + réponses       │    │  contextuelle)      │    │  priorisé             │
└──────────────────┘    └────────────────────┘    └─────────────────────┘
```

### Principe directeur : une architecture 100% agnostique du fournisseur LLM

Le point technique le plus important de ce projet : **aucune brique du code métier (RAG, routeur, scoring, API) ne connaît le nom d'un fournisseur LLM particulier.** Tout passe par une interface abstraite (`LLMProvider`), implémentée par 4 fournisseurs interchangeables :

| Provider | Cas d'usage | Configuration |
|---|---|---|
| `mock` | Développement/démo sans clé API, tests, CI | Aucune (par défaut) |
| `openai` | Production avec l'API OpenAI standard | `OPENAI_API_KEY` |
| `azure_openai` | Production sur stack Microsoft (Azure OpenAI Service) | `AZURE_OPENAI_*` |
| `ollama` | LLM local, 100% souverain, aucune donnée envoyée à un tiers | `OLLAMA_*` |

Changer de fournisseur = changer **une seule variable** (`LLM_PROVIDER`) dans `.env`. C'est un pattern "Strategy" classique en ingénierie logicielle, qui évite le vendor lock-in — argument que je peux développer en entretien technique.

La même logique s'applique aux embeddings (`src/rag/embeddings.py`) : `tfidf` (100% hors-ligne, aucun téléchargement, par défaut), `local` (sentence-transformers, meilleure qualité sémantique), ou `openai`.

---

## 📂 Structure du projet

```
governance-rag-copilot/
├── data/
│   └── knowledge_base/          # Référentiels de sécurité (JSON), reformulés (pas de copie verbatim)
│       ├── iso27001_controls.json
│       ├── nis2_requirements.json
│       └── rgpd_principles.json
├── src/
│   ├── config.py                 # Configuration centrale (tout surchargeable via .env)
│   ├── ingestion/
│   │   └── loader.py             # Chargement + chunking de la base de connaissances
│   ├── rag/
│   │   ├── llm_provider.py       # Abstraction LLM (mock/openai/azure_openai/ollama)
│   │   ├── embeddings.py         # Abstraction embeddings (tfidf/local/openai)
│   │   ├── vectorstore.py        # Wrapper ChromaDB
│   │   ├── retriever.py          # Recherche + seuil de pertinence
│   │   ├── intent_router.py      # Classification d'intention (règles + repli LLM)
│   │   └── chatbot.py            # Orchestrateur principal du chatbot
│   ├── scoring/
│   │   ├── models.py             # Modèles de données (contexte projet, réponses, résultats)
│   │   ├── criticality.py        # Logique de pondération contextuelle
│   │   └── engine.py             # Moteur d'évaluation de conformité
│   ├── reporting/
│   │   └── report_generator.py   # Génération de rapport Markdown avec plan priorisé
│   └── api/
│       └── main.py               # API FastAPI (/chat, /controls, /evaluate)
├── evaluation/
│   ├── test_questions.json       # Jeu de test pour l'évaluation du RAG
│   └── eval_rag.py                # Mesure Recall@k, Precision@k, groundedness
├── tests/
│   ├── test_scoring.py           # 21 tests unitaires du moteur de scoring
│   └── test_intent_router.py     # Tests du routeur d'intention
├── scripts/
│   ├── build_index.py            # Indexe la base de connaissances
│   └── demo.py                    # Démo end-to-end (chatbot + scoring)
├── requirements.txt
└── .env.example
```

---

## 🚀 Installation et démarrage rapide

```bash
git clone <votre-repo>
cd governance-rag-copilot

python3 -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env              # fonctionne tel quel, aucune clé API requise

# 1. Indexer la base de connaissances
python scripts/build_index.py

# 2. Lancer la démo complète (chatbot + scoring)
python scripts/demo.py

# 3. (optionnel) Lancer l'API
uvicorn src.api.main:app --reload --port 8000
# puis http://localhost:8000/docs pour la doc interactive Swagger

# 4. Lancer les tests
pytest tests/ -v

# 5. Évaluer la qualité du RAG
python evaluation/eval_rag.py
```

**Aucune clé API n'est nécessaire pour tester le projet** : par défaut, `LLM_PROVIDER=mock` et `EMBEDDING_PROVIDER=tfidf` font tourner l'ensemble du pipeline (retrieval réel, scoring réel, seule la génération de texte final est simulée).

### Brancher un vrai LLM (ex. Azure OpenAI)

Dans `.env` :
```bash
LLM_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY=<votre-clé>
AZURE_OPENAI_ENDPOINT=https://<votre-ressource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=<nom-du-déploiement>
```
Aucune autre modification de code nécessaire.

---

## 📊 Résultats mesurés

### Qualité du retrieval RAG (`evaluation/eval_rag.py`)

Sur un jeu de test de 9 questions couvrant les 3 référentiels indexés :

| Métrique | Valeur |
|---|---|
| Recall@4 moyen (questions factuelles) | **0.875** |
| Precision@4 moyenne (questions factuelles) | **0.938** |
| Rejet correct des questions hors-sujet | **1 / 1** |

Le rejet correct des questions hors-sujet est un point de conception important : si aucun passage ne dépasse le seuil de similarité configuré (`RAG_SIMILARITY_THRESHOLD`), le système répond explicitement qu'il ne trouve pas d'exigence pertinente, **plutôt que de générer une réponse non sourcée**. C'est un choix délibéré pour un cas d'usage de conformité, où une réponse hallucinée peut avoir des conséquences réglementaires.

### Tests unitaires

25 tests couvrant : la logique de pondération contextuelle (facteurs cumulés selon le contexte du projet), la classification du niveau de risque global, le moteur de scoring de bout en bout, et le routeur d'intention basé sur des règles.

```bash
pytest tests/ -v
# 25 passed
```

---

## 🧠 Choix de conception à retenir

- **Traçabilité systématique** : chaque réponse du chatbot cite l'identifiant exact du contrôle utilisé (ex. `ISO27001-A.8.24`). Aucune affirmation sur une exigence de sécurité n'est faite sans source vérifiable.
- **Pondération contextuelle, pas un score plat** : le moteur de scoring ne traite pas tous les contrôles de la même façon quel que soit le projet évalué. Le facteur d'amplification (`ProjectContext.facteur_amplification()`) rend explicite et auditable pourquoi un contrôle pèse plus lourd dans tel contexte — plutôt qu'un modèle boîte noire.
- **Routeur d'intention hybride** : des règles simples et bon marché en calcul pour les cas évidents (mots-clés d'urgence, de demande d'évaluation), avec repli sur un classifieur LLM pour les cas ambigus — pattern courant en production pour limiter les coûts d'inférence.
- **Priorité systématique à l'escalade humaine** : en cas d'ambiguïté entre "demande d'évaluation" et "urgence/incident", le routeur privilégie toujours l'escalade humaine (principe de précaution).
- **Aucune dépendance à un fournisseur cloud unique** : tout le pipeline (LLM, embeddings, vector store) est interchangeable, ce qui permet de démontrer le projet sur n'importe quelle stack (OpenAI, Azure OpenAI, 100% local).

---

## ⚠️ Limites assumées et pistes d'amélioration

- La base de connaissances (19 contrôles) est un **jeu de démonstration**, rédigé dans mes propres mots à partir de la structure publique des référentiels — pas une copie intégrale des textes officiels (payants pour ISO 27001). En production, elle serait alimentée par les textes réglementaires complets sous licence appropriée.
- Le mode d'embeddings par défaut (`tfidf`) privilégie la robustesse hors-ligne à la qualité sémantique. Pour une similarité plus fine (synonymes, paraphrases), passer à `EMBEDDING_PROVIDER=local` (sentence-transformers) ou `openai`.
- Le classifieur d'intention en mode `mock` retombe sur `question_generale` par défaut (aucun vrai LLM appelé) : la classification LLM complète n'est démontrable qu'avec un provider réel configuré.
- Pistes d'extension : ajout d'un référentiel DORA ou PCI-DSS (il suffit d'ajouter un fichier JSON dans `data/knowledge_base/`, aucune modification de code requise) ; interface web (Streamlit) par-dessus l'API FastAPI ; connexion à Copilot Studio via un connecteur HTTP personnalisé pointant vers `/chat`.

---

## 🛠️ Stack technique

Python 3.12 · ChromaDB · scikit-learn (TF-IDF) · sentence-transformers (optionnel) · FastAPI · Pydantic · pytest · OpenAI SDK (compatible Azure OpenAI)

---

## Licence

Projet personnel à but pédagogique et de démonstration. Les extraits de référentiels de sécurité sont des reformulations synthétiques à des fins de démonstration et ne remplacent pas les textes réglementaires officiels.
