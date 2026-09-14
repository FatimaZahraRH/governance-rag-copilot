"""
API FastAPI exposant les deux briques du projet :
  - POST /chat            : interroge le chatbot de gouvernance
  - GET  /controls         : liste les contrôles disponibles (pour générer un questionnaire)
  - POST /evaluate         : lance une évaluation de conformité et retourne le rapport

Cette API peut être appelée depuis n'importe quel front (Streamlit, React,
Teams via un connecteur, ou même Copilot Studio via une "custom connector"
HTTP — ce qui la rend directement compatible avec l'écosystème Microsoft).

Lancement local :
    uvicorn src.api.main:app --reload --port 8000
    puis http://localhost:8000/docs pour la documentation interactive.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.chatbot import GovernanceChatbot
from src.rag.embeddings import get_embedding_provider
from src.rag.llm_provider import get_llm_provider
from src.rag.retriever import Retriever
from src.rag.vectorstore import VectorStore
from src.reporting.report_generator import generate_markdown_report
from src.scoring.engine import ComplianceScoringEngine
from src.scoring.models import ControlAnswer, NiveauReponse, ProjectContext

app = FastAPI(
    title="Governance RAG Copilot API",
    description="Chatbot de gouvernance cyber + moteur de scoring de conformité (architecture agnostique LLM).",
    version="1.0.0",
)


# --------------------------------------------------------------------------
# Dépendances mises en cache (évite de recharger le modèle d'embeddings et
# de rouvrir la connexion au vector store à chaque requête)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_chatbot() -> GovernanceChatbot:
    embedding_provider = get_embedding_provider()
    store = VectorStore(embedding_provider)
    retriever = Retriever(store)
    llm = get_llm_provider()
    return GovernanceChatbot(retriever, llm)


@lru_cache(maxsize=1)
def get_scoring_engine() -> ComplianceScoringEngine:
    return ComplianceScoringEngine()


# --------------------------------------------------------------------------
# Schémas Pydantic (contrat d'API)
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["Quelles sont les exigences de chiffrement ?"])


class SourceOut(BaseModel):
    chunk_id: str
    referentiel: str
    titre: str
    similarity: float


class ChatResponseOut(BaseModel):
    answer: str
    intent: str
    intent_confidence: float
    grounded: bool
    sources: List[SourceOut]


class ProjectContextIn(BaseModel):
    nom_projet: str
    traite_donnees_personnelles: bool = False
    traite_donnees_sensibles: bool = False
    expose_sur_internet: bool = False
    fournisseur_tiers: bool = False
    volumetrie_utilisateurs: str = "faible"


class ControlAnswerIn(BaseModel):
    control_id: str
    reponse: str  # "conforme" | "partiel" | "non_conforme" | "non_applicable"
    commentaire: Optional[str] = ""


class EvaluationRequest(BaseModel):
    project_context: ProjectContextIn
    answers: List[ControlAnswerIn]


class EvaluationResponseOut(BaseModel):
    score_global_pourcent: float
    niveau_risque_global: str
    facteur_amplification: float
    rapport_markdown: str


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponseOut)
def chat(request: ChatRequest) -> ChatResponseOut:
    chatbot = get_chatbot()
    response = chatbot.ask(request.message)
    return ChatResponseOut(
        answer=response.answer,
        intent=response.intent.intent.value,
        intent_confidence=response.intent.confidence,
        grounded=response.grounded,
        sources=[
            SourceOut(chunk_id=s.chunk_id, referentiel=s.referentiel, titre=s.titre, similarity=s.similarity)
            for s in response.sources
        ],
    )


@app.get("/controls")
def list_controls() -> List[dict]:
    engine = get_scoring_engine()
    return engine.list_available_controls()


@app.post("/evaluate", response_model=EvaluationResponseOut)
def evaluate(request: EvaluationRequest) -> EvaluationResponseOut:
    engine = get_scoring_engine()

    try:
        context = ProjectContext(**request.project_context.model_dump())
        answers = [
            ControlAnswer(control_id=a.control_id, reponse=NiveauReponse(a.reponse), commentaire=a.commentaire or "")
            for a in request.answers
        ]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    report = engine.evaluate(context, answers)
    markdown = generate_markdown_report(report, engine)

    return EvaluationResponseOut(
        score_global_pourcent=report.score_global_pourcent,
        niveau_risque_global=report.niveau_risque_global,
        facteur_amplification=context.facteur_amplification(),
        rapport_markdown=markdown,
    )
