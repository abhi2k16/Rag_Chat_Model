# RAG Chat Model

This folder contains a local Retrieval-Augmented Generation (RAG) project built with LangChain, Ollama, HuggingFace embeddings, PDF parsing, and optional PostgreSQL/PGVector storage.

The main execution file is `generation_rag.py`. It loads PDF documents, splits them into chunks, builds vector stores, retrieves relevant context, and uses an Ollama chat model to generate answers from the retrieved document content.

## Project Introduction

This project builds a local RAG chatbot for asking questions over PDF documents. It indexes documents such as research papers or lecture notes, converts their text into searchable vector embeddings, retrieves the most relevant chunks for a user question, and sends that context to a local LLM to generate a grounded answer.

Work completed in this project includes:

- PDF document loading and text extraction.
- Text cleaning and chunking for better retrieval.
- Document change tracking with stable document IDs.
- HuggingFace embedding generation.
- In-memory vector search.
- Optional persistent vector storage with PostgreSQL/PGVector.
- Retrieval testing with similarity, MMR, and filtered search.
- Interactive RAG answer generation with prompt switching, streaming, batch mode, debug mode, and prompt preview.

Main libraries and their use in this project:

| Library | Use in this project |
| --- | --- |
| `LangChain` | Connects the full RAG workflow: document objects, text splitting, retrievers, prompt templates, chains, and output parsing. |
| `Ollama` | Runs the local LLM used for answer generation, mainly `llama3.2:1b` in `generation_rag.py`. |
| `HuggingFace` | Provides the local embedding model `sentence-transformers/all-MiniLM-L6-v2` for converting text chunks and queries into vectors. |
| `pdfplumber` | Extracts text from PDF pages before indexing. |
| `InMemoryVectorStore` | Stores document vectors in RAM for fast local retrieval. |
| `PGVector` | Optionally stores vectors in PostgreSQL for persistent retrieval. |
| `SQLRecordManager` | Tracks indexed chunks and helps avoid duplicate or stale indexing. |

## Main Workflow

When you run the main executable:

```powershell
python main.py
```

the pipeline runs in this order:

1. Load project configuration and helper functions from `IndexingDocs_for_rag.py`.
2. Load document ingestion and vector-store setup functions from `retrieval_rag.py`.
3. Read every PDF file in `rag_docs/`.
4. Extract and clean text from each PDF page.
5. Split pages into overlapping text chunks.
6. Create HuggingFace embeddings for the chunks.
7. Store vectors in `InMemoryVectorStore`.
8. Try to store vectors in PostgreSQL/PGVector.
9. Build a retriever, preferring PGVector if available.
10. Initialize the local Ollama LLM.
11. Start an interactive RAG question-answering loop.

The generation chain is:

```text
User question
  -> retriever
  -> relevant document chunks
  -> format_docs()
  -> prompt template
  -> ChatOllama
  -> StrOutputParser
  -> final answer
```

## File Responsibilities

### `generation_rag.py`

This is the main RAG generation entry point.

It is responsible for:

- Starting the full RAG pipeline.
- Selecting and switching prompt templates.
- Building the final LangChain generation chain.
- Running standard, streaming, and batch generation.
- Showing rendered prompt previews.
- Debugging retrieved chunks before generation.
- Running the interactive terminal loop.

Important imports from `IndexingDocs_for_rag.py`:

| Import | Purpose |
| --- | --- |
| `PDF_FILES` | List of PDF files to index. |
| `RECORD_DB` | SQLite record-manager database path. |
| `HF_MODEL` | HuggingFace embedding model name. |
| `PG_CONNECTION` | PostgreSQL/PGVector connection string. |
| `COLLECTION_NAME` | PGVector collection name. |
| `clean_text()` | Cleans extracted PDF text. |
| `load_tracker()` | Loads document tracking metadata. |
| `get_or_create_doc_id()` | Reuses or creates stable document IDs. |
| `build_embeddings()` | Creates the HuggingFace embedding model. |
| `build_vectorstore()` | Creates the in-memory vector store. |
| `build_pg_vectorstore()` | Creates the PGVector store if PostgreSQL is available. |
| `build_retriever()` | Chooses PGVector or InMemory retriever. |
| `build_prompt()` | Creates prompt templates by type. |
| `format_docs()` | Formats retrieved chunks into prompt context. |

Important imports from `retrieval_rag.py`:

| Import | Purpose |
| --- | --- |
| `load_and_split_docs()` | Loads PDFs, cleans pages, splits text into chunks, and assigns chunk IDs. |
| `build_stores()` | Builds embeddings, InMemoryVectorStore, and optional PGVector storage. |

### `retrieval_rag.py`

This module handles the retrieval stage. It is imported by `generation_rag.py`, but it can also be run directly for retrieval testing.

It is responsible for:

- Loading PDFs with `pdfplumber`.
- Creating LangChain `Document` objects with metadata.
- Splitting text with `RecursiveCharacterTextSplitter`.
- Assigning `chunk_id` metadata.
- Building HuggingFace embeddings.
- Indexing chunks into `InMemoryVectorStore`.
- Attempting PGVector storage.
- Running similarity, MMR, and filtered searches.

Run it directly with:

```powershell
python retrieval_rag.py
```

This is useful when you want to test retrieval quality without focusing on prompt switching or generation commands.

### `IndexingDocs_for_rag.py`

This is the shared configuration and helper module used by both retrieval and generation.

It defines:

- PDF paths.
- Tracker file path.
- SQLite record-manager path.
- HuggingFace embedding model.
- PostgreSQL/PGVector connection settings.
- Text cleaning logic.
- File hash and document change tracking.
- Vector-store builders.
- Prompt builders.
- Retriever builders.
- Default RAG chain builder.

When executed directly:

```powershell
python IndexingDocs_for_rag.py
```

it runs a full indexing demonstration:

1. Checks document changes using file hashes.
2. Loads PDFs.
3. Splits text into chunks.
4. Builds embeddings.
5. Indexes into InMemoryVectorStore.
6. Optionally indexes into PGVector.
7. Runs vector-store operations and semantic search.
8. Starts a basic RAG Q&A loop.

### `IndexingData_for_rag.py`

This is an earlier single-document RAG example using Ollama embeddings.

It:

- Loads `Attention_is_All_You_Need.pdf`.
- Splits it into chunks.
- Creates embeddings with `OllamaEmbeddings(model="llama3")`.
- Stores chunks in `InMemoryVectorStore`.
- Optionally stores chunks in PGVector.
- Builds a simple RAG chain.
- Starts an interactive Q&A loop.

Use this file as a simpler learning/demo script, not the main pipeline.

### `IndexingData_for_rag_langchain_ollama.py`

This is a multi-document indexing and RAG example using Ollama embeddings.

It adds:

- Multiple PDF support.
- Document change tracking.
- Stable document IDs.
- SQLite-backed `SQLRecordManager`.
- Incremental indexing.
- Optional PGVector storage.

Use this file to understand the older Ollama-embedding version of the indexing workflow.

### `ollama_rag.py`

This is a compact RAG script for a single PDF.

It demonstrates:

- PDF loading.
- Chunking.
- Ollama embeddings.
- In-memory retrieval.
- Prompt creation.
- Ollama generation.
- Interactive Q&A.

Use this as the simplest end-to-end RAG reference.

### `ollama_chat.py`

This is a basic Ollama chat test script.

It demonstrates:

- `ChatOllama` with system and human messages.
- `OllamaLLM` with a plain prompt.

Use this to verify that Ollama and LangChain can communicate before running the full RAG pipeline.

## Runtime and Data Files

| File or folder | Purpose |
| --- | --- |
| `Attention_is_All_You_Need.pdf` | Source PDF used for RAG. |
| `MachineLearning-Lecture01.pdf` | Source PDF used in older/demo workflows. |
| `rag_docs/` | Folder for additional RAG documents, if used by your experiments. |
| `doc_change_tracker.json` | Tracker for older Ollama-embedding indexing workflow. |
| `hf_doc_change_tracker.json` | Tracker for HuggingFace-embedding workflow. |
| `record_manager.db` | SQLite record-manager database for older workflow. |
| `hf_record_manager.db` | SQLite record-manager database for HuggingFace workflow. |

The tracker JSON files store file hashes, stable document IDs, indexing timestamps, and chunk counts. The SQLite databases are used by LangChain `SQLRecordManager` to avoid duplicate indexing and support incremental cleanup.

## Running the Main RAG Model

### 1. Start Ollama

Make sure Ollama is installed and running locally.

Pull the model used by `generation_rag.py`:

```powershell
ollama pull llama3.2:1b
```

Optional older scripts use `llama3`:

```powershell
ollama pull llama3
```

### 2. Install Python dependencies

The project uses packages such as:

- `langchain`
- `langchain-core`
- `langchain-ollama`
- `langchain-huggingface`
- `langchain-postgres`
- `langchain-text-splitters`
- `langchain-classic`
- `pdfplumber`
- `sentence-transformers`
- `psycopg`
- `tqdm`

Install them in your active Python environment if they are missing.

### 3. Check PDF paths

The active configuration is inside `IndexingDocs_for_rag.py`.

PDF documents are loaded automatically from:

```text
rag_docs/
```

Check these values before running:

```python
PDF_FOLDER
PDF_FILES
TRACKER_FILE
RECORD_DB
PG_CONNECTION
COLLECTION_NAME
```

If PDFs are moved, put them in `rag_docs/` or update `PDF_FOLDER` in `IndexingDocs_for_rag.py`.

### 4. Run generation

```powershell
python main.py
```

After setup, the terminal shows:

```text
Generation Ready! Commands:
  prompt
  preview:<query>
  debug:<query>
  stream:<query>
  batch:<q1>|<q2>
  <query>
  exit
```

## Interactive Commands in `generation_rag.py`

| Command | Description |
| --- | --- |
| `prompt` | Opens a menu to switch prompt type. |
| `preview:<query>` | Retrieves context and shows the rendered prompt without calling the LLM. |
| `debug:<query>` | Shows retrieved chunks, filenames, pages, and sample text. |
| `stream:<query>` | Streams the generated answer token by token. |
| `batch:<q1>|<q2>` | Runs multiple questions in one batch call. |
| `<query>` | Runs normal retrieve-and-generate answering. |
| `exit` or `quit` | Stops the interactive loop. |

Example queries:

```text
What is the Transformer architecture?
debug:What is multi-head attention?
preview:Explain self-attention
stream:Summarize positional encoding
batch:What is attention?|What is the encoder?
```

## Prompt Types

`generation_rag.py` supports these prompt modes:

| Mode | Description |
| --- | --- |
| `default` | Standard helpful assistant answer. |
| `concise` | Short 1-2 sentence answer. |
| `detailed` | Detailed answer with document/page references. |
| `bullet` | Bullet-point answer. |
| `custom` | User writes a custom prompt containing `{context}` and `{question}`. |

Prompt templates are created by `build_prompt()` in `IndexingDocs_for_rag.py`.

## Vector Store Behavior

The project uses two vector-store options:

### InMemoryVectorStore

- Always attempted.
- Stores vectors in RAM.
- Fast and simple.
- Rebuilt each run.
- Does not persist vectors after the Python process exits.

### PGVector

- Optional.
- Requires PostgreSQL with the vector extension.
- Stores vectors persistently.
- Used automatically if `build_pg_vectorstore()` connects successfully.
- If unavailable, the pipeline falls back to `InMemoryVectorStore`.

`build_retriever()` chooses PGVector first when available:

```text
PGVector available -> use PGVector retriever
PGVector unavailable -> use InMemoryVectorStore retriever
```

## Document Metadata

Each loaded page or chunk carries metadata:

| Metadata key | Meaning |
| --- | --- |
| `source` | Full PDF path. |
| `filename` | PDF filename. |
| `doc_id` | Stable UUID for the source document. |
| `page` | Source page number. |
| `chunk_id` | Unique chunk identifier. |

This metadata is used for traceability in prompt context, debug output, and filtered retrieval.

## Notes

- `generation_rag.py` is the recommended file for the complete interactive RAG generation workflow.
- `retrieval_rag.py` is useful for retrieval testing and search experiments.
- `IndexingDocs_for_rag.py` contains the shared configuration and reusable builders.
- `IndexingData_*` and `ollama_*` files are useful learning/demo scripts, but they are not the main active generation workflow.
- PGVector is optional. The project still runs with only InMemoryVectorStore if PostgreSQL is not available.
