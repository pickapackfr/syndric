from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import streamlit as st
from utils import convert_to_chat_messages


# Settings control global defaults
Settings.embed_model = OllamaEmbedding(model_name="qwen3-embedding:0.6b")
Settings.llm = Ollama(
    model="qwen3:latest",
    request_timeout=60.0,
    context_window=4000,
)

st.header("Syndric v0.1-turbo 🏢🤖")

# Initialize the first message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": """Tu es un assistant IA destiné au syndic de copropriété appellé Syndric. 
            Tu aides les utilisateurs en répondant à leurs questions de manière sympathique et empathique. 
            Tu dois toujours répondre en français. La monnaie utilisée est l'euro (€).""",
        },
        {
            "role": "assistant",
            "content": """Bonjour, je suis Syndric, votre assistant d'IA sympathique ! Comment puis-je vous aider aujourd'hui ?""",
        },
    ]


@st.cache_resource(show_spinner=True)
def load_syndic_data():
    docs = SimpleDirectoryReader(input_dir="data", recursive=True).load_data()
    vectorStoreIndex = VectorStoreIndex.from_documents(docs, show_progress=True)
    return vectorStoreIndex


index = load_syndic_data()

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context", verbose=False, streaming=True
    )

# Prompt for user input and save to chat history
if prompt := st.chat_input("Votre question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display all chat messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Generate a response if the last message is from the user
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        if any(
            keyword.lower() in prompt.lower()
            for keyword in [
                "syndic",
                "copropriété",
                "charges",
                "immeuble",
                "résidents",
                "assemblée",
                "budget",
                "travaux",
                "entretien",
                "alexandre",
                "perez",
            ]
        ):
            with st.spinner("Recherche dans les données du syndic..."):
                response_stream = st.session_state.chat_engine.stream_chat(prompt)
                st.write_stream(response_stream.response_gen)
                message = {"role": "assistant", "content": response_stream.response}
                st.session_state.messages.append(message)
        else:
            with st.spinner("Génération de la réponse..."):
                response_stream = Settings.llm.stream_chat(
                    convert_to_chat_messages(st.session_state.messages)
                )
                response = ""
                for chunk in response_stream:
                    response += chunk.delta
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)
