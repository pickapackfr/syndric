# Syndrik IA

A friendly and empathetic AI assistant chatbot built with Streamlit, LlamaIndex, and Ollama.

## Overview

Syndrik is an AI-powered chat application that allows users to interact with a local language model. It processes documents from a local directory, builds a vector index for semantic search, and responds to user queries in French using a conversational interface.

**Key Features:**

- Local LLM using Ollama (no external API keys needed)
- Document processing and semantic search via LlamaIndex
- Interactive chat interface with Streamlit
- Conversation history management
- Streaming responses for real-time user feedback
- French language support

## Architecture

- **LLM**: Ollama with `qwen3:latest` model
- **Embedding Model**: Ollama `qwen3-embedding:0.6b`
- **Framework**: Streamlit (web UI)
- **Search**: LlamaIndex with vector index

## Prerequisites

### System Requirements

- Python ≥ 3.14
- Ollama installed and running locally

### Install Ollama

1. Download from [ollama.ai](https://ollama.ai)
2. Install and start the Ollama service
3. Pull the required models:

```bash
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b
```

Verify Ollama is running (default: `http://localhost:11434`):

```bash
curl http://localhost:11434/api/tags
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pickapackfr/syndric.git
cd syndric
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # On Windows PowerShell
# or
source venv/bin/activate     # On macOS/Linux
```

### 3. Install Dependencies

```bash
uv sync
```

This installs the project in editable mode with dependencies from `pyproject.toml`.

### 4. Prepare Data

Create a `data` directory in the project root and add your documents:

```bash
mkdir data
# Copy your text or PDF files to the data/ directory
```

The application will recursively read all files from this directory.

## Running the Application

Start the Streamlit app:

```bash
streamlit run src/main.py
```

The app will open in your browser (typically `http://localhost:8501`).

### First Run

On the first run, the application will:

1. Read all documents from the `./data` directory
2. Generate embeddings using the Ollama embedding model
3. Build a vector index
4. Initialize the chat engine

This may take a few moments depending on the size of your documents.

## Usage

1. Enter your question in the chat input box (questions in French are preferred)
2. The assistant will:
   - Search the indexed documents for relevant context
   - Generate a response using the Ollama LLM
   - Stream the response in real-time
   - Add it to the conversation history

Example questions:

- _"Quel est le sujet principal du document ?"_ (What is the main subject of the document?)
- _"Peux-tu résumer ce contenu ?"_ (Can you summarize this content?)

## Project Structure

```
syndric/
├── src/
│   ├── main.py                 # Streamlit chat application
│   ├── backup.py               # Backup utilities
│   └── externals/
│       ├── ollama.py           # Ollama integration reference
│       ├── openai.py           # OpenAI integration (optional)
│       └── minio.py            # MinIO storage integration (optional)
├── tests/
│   ├── __init__.py
│   └── externals/
│       └── test_minio.py       # Integration tests
├── data/                       # Place your documents here
├── pyproject.toml              # Project dependencies
└── README.md                   # This file
```

## Configuration

Edit `src/main.py` to customize:

- **Embedding Model**: Change `model_name` in `OllamaEmbedding(model_name="qwen3-embedding:xxx")`
- **LLM Model**: Change `model="qwen3:xxx"` in the `Ollama()` constructor
- **Context Window**: Adjust `context_window=4000` (in tokens)
- **System Prompt**: Modify the `system_prompt` parameter (currently in French)
- **Data Directory**: Change `input_dir="./data"` in `SimpleDirectoryReader()`

## Troubleshooting

### "Connection refused" to Ollama

- Ensure Ollama is running: `ollama serve` or check your system tray
- Verify models are pulled: `ollama list`
- Check Ollama is listening on `http://localhost:11434`

### Out of memory errors

- Reduce the `context_window` in `main.py`
- Use a smaller model variant
- Reduce the amount of data in the `./data` directory

### Slow responses

- Ensure sufficient system RAM
- Close other applications
- Consider using a faster CPU or GPU

## Dependencies

Key packages (see `pyproject.toml` for full list):

- `llama-index` — Document indexing and search
- `streamlit` — Web UI framework
- `ollama` — Local LLM inference

## License

Team 5 Project

## Support

For issues or questions, refer to the [LlamaIndex docs](https://docs.llamaindex.ai) and [Streamlit docs](https://docs.streamlit.io).
