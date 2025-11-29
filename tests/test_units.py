import pytest
from unittest.mock import Mock, patch, MagicMock
from llama_index.core.llms import ChatMessage, MessageRole
from src.utils import convert_to_chat_messages, route_message


class TestConvertToChatMessages:
    """Tests pour la fonction convert_to_chat_messages"""

    def test_convert_empty_list(self):
        """Test avec une liste vide"""
        messages = []
        result = convert_to_chat_messages(messages)
        assert result == []

    def test_convert_single_user_message(self):
        """Test avec un seul message utilisateur"""
        messages = [{"role": "user", "content": "Hello"}]
        result = convert_to_chat_messages(messages)

        assert len(result) == 1
        assert result[0].role == MessageRole.USER
        assert result[0].content == "Hello"

    def test_convert_single_assistant_message(self):
        """Test avec un seul message assistant"""
        messages = [{"role": "assistant", "content": "Hi there!"}]
        result = convert_to_chat_messages(messages)

        assert len(result) == 1
        assert result[0].role == MessageRole.ASSISTANT
        assert result[0].content == "Hi there!"

    def test_convert_single_system_message(self):
        """Test avec un seul message système"""
        messages = [{"role": "system", "content": "You are a helpful assistant"}]
        result = convert_to_chat_messages(messages)

        assert len(result) == 1
        assert result[0].role == MessageRole.SYSTEM
        assert result[0].content == "You are a helpful assistant"

    def test_convert_multiple_messages(self):
        """Test avec plusieurs messages de différents rôles"""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Answer 2"},
        ]
        result = convert_to_chat_messages(messages)

        assert len(result) == 5
        assert result[0].role == MessageRole.SYSTEM
        assert result[0].content == "System prompt"
        assert result[1].role == MessageRole.USER
        assert result[1].content == "Question 1"
        assert result[2].role == MessageRole.ASSISTANT
        assert result[2].content == "Answer 1"
        assert result[3].role == MessageRole.USER
        assert result[3].content == "Question 2"
        assert result[4].role == MessageRole.ASSISTANT
        assert result[4].content == "Answer 2"

    def test_convert_preserves_order(self):
        """Test que l'ordre des messages est préservé"""
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        result = convert_to_chat_messages(messages)

        assert result[0].content == "First"
        assert result[1].content == "Second"
        assert result[2].content == "Third"

    def test_convert_with_special_characters(self):
        """Test avec des caractères spéciaux dans le contenu"""
        messages = [
            {"role": "user", "content": "Quel est le montant des charges?"},
            {"role": "assistant", "content": "Les charges sont de 150€/mois"},
        ]
        result = convert_to_chat_messages(messages)

        assert len(result) == 2
        assert result[0].content == "Quel est le montant des charges?"
        assert result[1].content == "Les charges sont de 150€/mois"


class TestRouteMessage:
    """Tests pour la fonction route_message"""

    @patch("src.utils.Ollama")
    def test_route_property_question(self, mock_ollama_class):
        """Test routage vers 'property' pour une question immobilière"""
        # Mock du LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "property"
        mock_llm.complete.return_value = mock_response
        mock_ollama_class.return_value = mock_llm

        messages = [
            {
                "role": "user",
                "content": "Quel est le montant des charges de copropriété?",
            }
        ]

        result = route_message(messages)

        assert result == "property"
        mock_llm.complete.assert_called_once()

    @patch("src.utils.Ollama")
    def test_route_general_question(self, mock_ollama_class):
        """Test routage vers 'general' pour une question générale"""
        # Mock du LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "general"
        mock_llm.complete.return_value = mock_response
        mock_ollama_class.return_value = mock_llm

        messages = [{"role": "user", "content": "Quelle est la capitale de la France?"}]

        result = route_message(messages)

        assert result == "general"
        mock_llm.complete.assert_called_once()

    @patch("src.utils.Ollama")
    def test_route_with_conversation_history(self, mock_ollama_class):
        """Test routage avec historique de conversation (utilise le dernier message utilisateur)"""
        # Mock du LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "property"
        mock_llm.complete.return_value = mock_response
        mock_ollama_class.return_value = mock_llm

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Bonjour"},
            {"role": "assistant", "content": "Bonjour! Comment puis-je vous aider?"},
            {"role": "user", "content": "Parle-moi du budget de la copropriété"},
        ]

        result = route_message(messages)

        assert result == "property"
        # Vérifie que le prompt contient le dernier message utilisateur
        call_args = mock_llm.complete.call_args[0][0]
        assert "Parle-moi du budget de la copropriété" in call_args

    @patch("src.utils.Ollama")
    def test_route_empty_messages(self, mock_ollama_class):
        """Test routage avec liste de messages vide"""
        messages = []

        result = route_message(messages)

        # Devrait retourner "general" par défaut
        assert result == "general"

    @patch("src.utils.Ollama")
    def test_route_no_user_message(self, mock_ollama_class):
        """Test routage sans message utilisateur"""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Assistant message"},
        ]

        result = route_message(messages)

        # Devrait retourner "general" par défaut
        assert result == "general"

    @patch("src.utils.Ollama")
    def test_route_llm_error_fallback(self, mock_ollama_class):
        """Test fallback vers 'general' en cas d'erreur du LLM"""
        # Mock du LLM qui lève une exception
        mock_llm = Mock()
        mock_llm.complete.side_effect = Exception("LLM error")
        mock_ollama_class.return_value = mock_llm

        messages = [{"role": "user", "content": "Test question"}]

        result = route_message(messages)

        # Devrait fallback vers "general"
        assert result == "general"

    @patch("src.utils.Ollama")
    def test_route_ollama_initialization(self, mock_ollama_class):
        """Test que Ollama est initialisé correctement"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "property"
        mock_llm.complete.return_value = mock_response
        mock_ollama_class.return_value = mock_llm

        messages = [{"role": "user", "content": "Question sur les charges"}]

        route_message(messages)

        # Vérifie les paramètres d'initialisation d'Ollama
        mock_ollama_class.assert_called_once_with(
            model="qwen3:4b",
            request_timeout=30.0,
            context_window=1000,
        )

    @patch("src.utils.Ollama")
    def test_route_syndic_keywords(self, mock_ollama_class):
        """Test avec différents mots-clés liés au syndic"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "property"
        mock_llm.complete.return_value = mock_response
        mock_ollama_class.return_value = mock_llm

        test_cases = [
            "Quels sont les travaux prévus?",
            "Quand est la prochaine assemblée générale?",
            "Quel est le règlement de copropriété?",
            "Qui sont les locataires?",
            "Y a-t-il des contrats d'entretien?",
        ]

        for question in test_cases:
            messages = [{"role": "user", "content": question}]
            result = route_message(messages)
            # Le mock retourne toujours "property"
            assert result == "property"

    @patch("src.utils.Ollama")
    def test_route_response_normalization(self, mock_ollama_class):
        """Test que la réponse est retournée telle quelle"""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "property"
        mock_llm.complete.return_value = mock_response
        mock_ollama_class.return_value = mock_llm

        messages = [{"role": "user", "content": "Test"}]
        result = route_message(messages)

        # La fonction retourne response.text directement
        assert result == "property"
