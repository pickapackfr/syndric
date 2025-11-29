from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Settings
from typing import Literal
from llama_index.llms.ollama import Ollama


def convert_to_chat_messages(messages):
    """Convertit les messages Streamlit en ChatMessage LlamaIndex"""
    chat_messages = []
    for msg in messages:
        role_map = {
            "system": MessageRole.SYSTEM,
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
        }
        chat_messages.append(
            ChatMessage(role=role_map[msg["role"]], content=msg["content"])
        )
    return chat_messages


def route_message(messages: list[dict]) -> Literal["property", "general"]:
    """Route le message utilisateur vers la base de données vectorielle ou le LLM général.

    Utilise le LLM pour classifier la dernière question de l'utilisateur comme:
    - "property": questions sur la gestion immobilière, copropriété, syndic
    - "general": questions générales sans rapport avec l'immobilier

    Args:
        messages: Liste des messages de la conversation Streamlit

    Returns:
        "property" ou "general" selon la classification
    """
    # Utilisation d'un modèle LLM pour la classification
    small_llm = Ollama(
        model="qwen3:4b",
        request_timeout=30.0,
        context_window=1000,
    )

    # Extraire la dernière question de l'utilisateur
    last_user_message = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_user_message = msg["content"]
            break

    if not last_user_message:
        # Fallback: si pas de message utilisateur, router vers property par défaut
        return "general"

    # Prompt de routage pour le LLM
    routing_prompt = f"""Tu agis comme un routeur de requêtes pour un assistant de syndic de copropriété.
        Analyse UNIQUEMENT la question suivante de l'utilisateur.
        Si la question concerne la gestion d'un appartement, d'une maison ou d'une copropriété (charges, travaux, syndic, règlement, assemblées générales, budgets, contrats, gestion des résidents, propriétaires, locataires, entretien d'immeuble, etc.), réponds par "property".
        Si la question est générale et ne concerne pas ce contexte immobilier/copropriété, réponds par "general".
        Réponds uniquement par "property" ou "general", sans aucune explication supplémentaire.
        Question de l'utilisateur: {last_user_message}
        Réponse:"""

    try:
        # Appeler le LLM pour classification (non-streaming)
        response = small_llm.complete(routing_prompt)

        # Normaliser la réponse
        return response.text

    except Exception as e:
        # En cas d'erreur, fallback vers property
        print(f"Erreur de routage: {e}")
        return "general"
