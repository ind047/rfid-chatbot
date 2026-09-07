# RFID Coding Assistant

A Retrieval-Augmented Generation (RAG) chatbot that lets you ask questions about the
[zetavg/Inventory](https://github.com/zetavg/Inventory) RFID inventory management software —
its architecture, code, and how things work. The app indexes the source code into a local
vector database and answers questions using a locally running LLM via Ollama. No code or
questions ever leave your machine.

## How It Works

1. **`ingest.py`** clones the RFID Inventory repository and indexes its source files
   (Java, TypeScript, JavaScript, Markdown) into a Chroma vector store using
   `all-MiniLM-L6-v2` sentence embeddings.
2. **`app.py`** runs a Streamlit chat UI. Each question retrieves the 5 most relevant
   code chunks from the vector store and passes them as context to a local LLM
   (`qwen2.5-coder:14b` via Ollama), which generates an answer with source file references.

```
Question → Chroma vector store (top-5 chunks) → Ollama LLM → Answer + Sources
```

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running, with the model pulled:
  ```bash
  ollama pull qwen2.5-coder:14b
  ```
- Git

## Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install streamlit langchain langchain-community langchain-chroma \
    langchain-huggingface langchain-ollama langchain-text-splitters

# (Optional) Use a different model or embedding model by editing the
# LLM_MODEL / EMBEDDING_MODEL constants in app.py and ingest.py
```

## Usage

**1. Build the knowledge base** (run once, or re-run after the upstream repo changes):

```bash
python ingest.py
```

This clones the Inventory repository into `rfid_repo/` and builds the vector store in
`chroma_db/`. It can take a few minutes depending on your machine.

**2. Start the chatbot:**

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501) and start asking
questions, e.g. *"How does the RFID tag scanning work?"* or *"Where is the inventory
database schema defined?"* Expandable **Sources** sections show which files the answer
was based on.

## Project Structure

```
├── app.py        # Streamlit chat UI + RAG chain (retrieval + generation)
├── ingest.py     # Clones the repo and builds the Chroma vector store
├── rfid_repo/    # Cloned Inventory repository (created by ingest.py, git-ignored)
├── chroma_db/    # Persistent vector store (created by ingest.py, git-ignored)
└── venv/         # Python virtual environment (git-ignored)
```

## Configuration

Key settings live as constants at the top of each script:

| Constant | Location | Default | Purpose |
|---|---|---|---|
| `REPO_URL` | `ingest.py` | `https://github.com/zetavg/Inventory.git` | Repository to index |
| `EMBEDDING_MODEL` | both | `all-MiniLM-L6-v2` | Sentence-transformer embedding model |
| `LLM_MODEL` | `app.py` | `qwen2.5-coder:14b` | Ollama model used to generate answers |
| `CHROMA_DB_DIR` | both | `./chroma_db` | Vector store location |

To point the assistant at a different codebase, change `REPO_URL` and delete `chroma_db/`
and `rfid_repo/` before re-running `ingest.py`.

## Notes

- The embedding model (`all-MiniLM-L6-v2`) is downloaded automatically on first run from
  Hugging Face and cached locally.
- The LLM must be available in Ollama — check with `ollama list`. A smaller model like
  `qwen2.5-coder:7b` works too if your hardware is limited.
