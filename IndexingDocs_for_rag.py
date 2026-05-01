import os                          # OS-level operations: env vars, file sizes, path joins
import re                          # Regular expressions for text cleaning
import json                        # Read/write the JSON change tracker file
import hashlib                     # MD5 hashing to detect file content changes
import uuid                        # Generate unique stable document IDs
from pathlib import Path           # Cross-platform file path handling
from datetime import datetime      # ISO timestamps for the tracker
from collections import Counter    # Count chunk distribution per document

# Remove SSL_CERT_FILE if it points to a missing file — prevents httpx/Ollama startup errors
ssl_cert_file = os.environ.get("SSL_CERT_FILE")
if ssl_cert_file and not Path(ssl_cert_file).is_file():
    os.environ.pop("SSL_CERT_FILE", None)

import pdfplumber                                              # Extract text from PDF pages
from langchain_huggingface import HuggingFaceEmbeddings        # Local HuggingFace embedding model, no API key needed
from langchain_ollama import ChatOllama                        # Ollama LLM wrapper (runs locally)
from langchain_core.vectorstores import InMemoryVectorStore    # In-RAM vector store for fast retrieval
from langchain_core.documents import Document                  # LangChain document wrapper with page_content + metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits text into overlapping chunks
from langchain_core.prompts import ChatPromptTemplate          # Build structured prompt templates
from langchain_core.runnables import RunnablePassthrough       # Passes the question unchanged into the chain
from langchain_core.output_parsers import StrOutputParser      # Converts LLM output to a plain Python string
from langchain_classic.indexes import index, SQLRecordManager  # Incremental indexing with deduplication tracking
from langchain_postgres.vectorstores import PGVector           # PostgreSQL-backed persistent vector store

# ─────────────────────────────────────────────
# CONFIG  (imported by retrieval.py and generation.py)
# ─────────────────────────────────────────────
PDF_FOLDER      = "c:\\Users\\abhij\\Desktop\\GenAIwithLLMs\\LangChain_projects\\"  # Root folder for all project files
TRACKER_FILE    = os.path.join(PDF_FOLDER, "hf_doc_change_tracker.json")           # JSON file that stores doc hashes and last-indexed timestamps
RECORD_DB       = f"sqlite:///{PDF_FOLDER}hf_record_manager.db"                   # SQLite DB used by SQLRecordManager to track which chunks are indexed
HF_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"                        # Lightweight 384-dim embedding model, runs fully on CPU
PG_USER         = "Langchain"                                                      # PostgreSQL username
PG_PASSWORD     = "Langchain"                                                      # PostgreSQL password
PG_CONNECTION   = f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@localhost:6024/Langchain"  # Full PGVector connection string (port 6024 = Docker container)
COLLECTION_NAME = "hf_docs"                                                        # PGVector collection name for this project

PDF_FILES = [                                                                      # List of PDF files to load and index
    os.path.join(PDF_FOLDER, "Attention_is_All_You_Need.pdf"),
    os.path.join(PDF_FOLDER, "LangChain Chat with Your Data", "MachineLearning-Lecture01.pdf"),
]

# ─────────────────────────────────────────────
# UTILITIES  (imported by retrieval.py and generation.py)
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove garbled characters, repeated patterns and excessive whitespace."""
    text = re.sub(r'(>\w+<\s*){2,}', '', text)    # Remove repeated OCR/PDF artifacts like >dap< >dap<
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)   # Replace non-printable/non-ASCII chars with a space
    text = re.sub(r'[ \t]{2,}', ' ', text)         # Collapse multiple spaces or tabs into one
    text = re.sub(r'\n{3,}', '\n\n', text)         # Collapse 3+ consecutive newlines into 2
    return text.strip()                             # Remove leading/trailing whitespace

def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file to detect content changes across runs."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):  # Read file in 8KB blocks for memory efficiency
            hasher.update(chunk)
    return hasher.hexdigest()                           # Return hex string like 'a3f9c2d1...'

def load_tracker() -> dict:
    """Load persisted indexing metadata from the previous run, or return empty dict."""
    if Path(TRACKER_FILE).exists():
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)                         # Returns dict keyed by filename
    return {}                                           # First run — no tracker file yet

def save_tracker(tracker: dict):
    """Persist document hashes and indexing metadata to disk after each run."""
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)                 # Pretty-print JSON for readability

def check_document_changes(filepath: str, tracker: dict) -> tuple[bool, str]:
    """
    Compare the current file hash against the stored hash.
    Returns (has_changed, status) where status is 'NEW', 'MODIFIED', or 'UNCHANGED'.
    """
    current_hash = compute_file_hash(filepath)
    filename     = Path(filepath).name                  # Use only the filename as the tracker key
    if filename not in tracker:
        return True, "NEW"                              # File has never been indexed before
    elif tracker[filename]["hash"] != current_hash:
        return True, "MODIFIED"                         # File content has changed since last index
    return False, "UNCHANGED"                           # File is identical to last indexed version

def get_or_create_doc_id(filepath: str, tracker: dict) -> str:
    """
    Return the existing doc_id for a file if it was indexed before,
    otherwise generate a new UUID. Reusing the same doc_id keeps
    incremental indexing consistent across runs.
    """
    filename = Path(filepath).name
    if filename in tracker and "doc_id" in tracker[filename]:
        return tracker[filename]["doc_id"]              # Reuse stable ID from previous run
    return str(uuid.uuid4())                            # New file — generate a fresh UUID

def update_tracker(filepath: str, tracker: dict, num_chunks: int, doc_id: str):
    """
    Save the latest indexing result for a single PDF into the tracker.
    Preserves the previous hash and timestamp for change history.
    """
    filename = Path(filepath).name
    now      = datetime.now().isoformat()               # ISO 8601 timestamp e.g. '2025-01-01T10:00:00'
    previous = tracker.get(filename, {})                # Get previous entry if it exists
    tracker[filename] = {
        "doc_id"          : doc_id,                     # Stable UUID for this document
        "hash"            : compute_file_hash(filepath), # Current file hash
        "last_indexed"    : now,                        # When this indexing run happened
        "num_chunks"      : num_chunks,                 # How many chunks were created
        "previous_hash"   : previous.get("hash", None),          # Hash from the previous run
        "previous_indexed": previous.get("last_indexed", None),   # Timestamp from the previous run
    }
    save_tracker(tracker)                               # Write updated tracker to disk

def build_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize and return the HuggingFace embedding model.
    all-MiniLM-L6-v2 produces 384-dim normalized vectors and runs on CPU.
    normalize_embeddings=True ensures cosine similarity works correctly.
    """
    return HuggingFaceEmbeddings(
        model_name=HF_MODEL,
        model_kwargs={"device": "cpu"},                 # Run on CPU — no GPU required
        encode_kwargs={"normalize_embeddings": True},   # L2-normalize vectors for cosine similarity
    )

def build_vectorstore(embeddings: HuggingFaceEmbeddings) -> InMemoryVectorStore:
    """
    Create and return an empty InMemoryVectorStore.
    Vectors are stored in RAM — fast but not persistent across restarts.
    """
    return InMemoryVectorStore(embeddings)

def build_pg_vectorstore(splits: list, embeddings: HuggingFaceEmbeddings):
    """
    Embed all chunks and store them in PGVector (PostgreSQL).
    pre_delete_collection=True drops and recreates the collection on each run
    to ensure a clean fresh index. Returns None if PostgreSQL is unavailable.
    """
    try:
        pg_vs = PGVector.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection=PG_CONNECTION,
            use_jsonb=True,                             # Store metadata as JSONB for flexible filtering
            pre_delete_collection=True,                 # Drop existing collection before re-indexing
        )
        return pg_vs
    except Exception as e:
        print(f"  [SKIPPED] PGVector not available: {e}")
        return None                                     # Gracefully skip if DB is not running

def format_docs(docs: list) -> str:
    """
    Format a list of retrieved Document objects into a single context string
    for the LLM prompt. Each chunk is prefixed with its filename, page, and doc_id.
    """
    return "\n\n".join(
        f"[{d.metadata.get('filename','?')} | Page {d.metadata.get('page','?')} | doc_id: {d.metadata.get('doc_id','?')}]: {d.page_content}"
        for d in docs
    )

def build_rag_chain(retriever, llm):
    """
    Assemble the default RAG chain:
      retriever → format_docs → prompt → llm → StrOutputParser
    Uses the default prompt template (helpful assistant style).
    """
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
""")
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}  # Retrieve + format context
        | prompt          # Fill the prompt template with context and question
        | llm             # Send filled prompt to the LLM
        | StrOutputParser()  # Extract plain text from the LLM response
    )

def build_prompt(prompt_type: str = "default") -> ChatPromptTemplate:
    """
    Return a ChatPromptTemplate by type.
    Available types: 'default', 'concise', 'detailed', 'bullet'
    Falls back to 'default' if an unknown type is given.
    Imported by generation.py to support switchable prompt styles.
    """
    templates = {
        "default": """
You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
""",
        "concise": """
Answer the question in 1-2 sentences using only the context below.
If not found, say "Not found in documents."

Context:
{context}

Question: {question}
""",
        "detailed": """
You are an expert assistant. Provide a thorough, well-structured answer using only the context below.
Cite the document name and page number for each key point.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
""",
        "bullet": """
Answer the question using only the context below. Format your answer as bullet points.
If not found, say "Not found in documents."

Context:
{context}

Question: {question}
""",
    }
    template = templates.get(prompt_type, templates["default"])  # Fall back to default if type not found
    return ChatPromptTemplate.from_template(template)

def build_retriever(vectorstore, pg_vectorstore=None, k: int = 4):
    """
    Return the active retriever and its label.
    Prefers PGVector (persistent) over InMemoryVectorStore (in-RAM).
    k controls how many chunks are returned per query.
    """
    if pg_vectorstore:
        return pg_vectorstore.as_retriever(search_kwargs={"k": k}), "PGVector"
    return vectorstore.as_retriever(search_kwargs={"k": k}), "InMemoryVectorStore"

# ─────────────────────────────────────────────
# MAIN INDEXING PIPELINE  (runs only when executed directly)
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── STEP 0: DOCUMENT CHANGE TRACKING ──────────────────────────────────
    # Check each PDF against the stored hash to decide if re-indexing is needed.
    print("=" * 50)
    print("STEP 0: Document Change Tracking")
    print("=" * 50)

    tracker        = load_tracker()   # Load previous run metadata (hashes, timestamps, doc_ids)
    files_to_index = []               # Will hold (pdf_path, doc_id) tuples for all PDFs

    for pdf_path in PDF_FILES:
        filename            = Path(pdf_path).name
        has_changed, status = check_document_changes(pdf_path, tracker)  # Compare hash to stored value
        doc_id              = get_or_create_doc_id(pdf_path, tracker)    # Reuse or create stable UUID
        print(f"  File     : {filename}")
        print(f"  Doc ID   : {doc_id}")
        print(f"  Status   : {status}")
        if status == "NEW":
            print(f"  Action   : First time indexing.")
        elif status == "MODIFIED":
            prev = tracker[filename]
            print(f"  Prev indexed : {prev['last_indexed']} | Prev hash: {prev['hash']}")
            print(f"  Action   : Document changed — re-indexing.")
        else:
            prev = tracker[filename]
            print(f"  Last indexed : {prev['last_indexed']} | Chunks: {prev['num_chunks']}")
            print(f"  Action   : No changes — loading from tracker.")
        files_to_index.append((pdf_path, doc_id))  # Always add to list so RAG can use it
        print()

    # ── STEP 1: LOAD MULTIPLE DOCUMENTS ───────────────────────────────────
    # Open each PDF, extract text page by page, clean it, and wrap in Document objects.
    print("=" * 50)
    print("STEP 1: Loading Multiple Documents")
    print("=" * 50)

    docs = []  # Accumulates one Document per valid page across all PDFs
    for pdf_path, doc_id in files_to_index:
        filename   = Path(pdf_path).name
        file_size  = os.path.getsize(pdf_path)  # File size in bytes for the info display
        page_count = 0                           # Pages that passed the content filter
        char_count = 0                           # Total characters across valid pages
        start_idx  = len(docs)                   # Index of the first page from this PDF in docs[]

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()       # Extract raw text from the PDF page
                if text and text.strip():
                    text = clean_text(text)      # Remove garbled chars and normalize whitespace
                    if len(text) < 30:
                        continue                 # Skip near-empty pages (e.g. figure-only pages)
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source"  : pdf_path,   # Full file path for traceability
                            "filename": filename,   # Short filename for display
                            "doc_id"  : doc_id,     # Stable UUID linking chunk back to its PDF
                            "page"    : i + 1,      # 1-based page number
                        }
                    ))
                    page_count += 1
                    char_count += len(text)

        # Print a summary box for each loaded document
        print(f"  ┌{'─' * 46}")
        print(f"  ├ File name      : {filename}")
        print(f"  ├ Doc ID         : {doc_id}")
        print(f"  ├ File path      : {pdf_path}")
        print(f"  ├ File size      : {file_size / 1024:.1f} KB ({file_size:,} bytes)")
        print(f"  ├ Total pages    : {total_pages}")
        print(f"  ├ Pages w/ text  : {page_count}")
        print(f"  ├ Total chars    : {char_count:,}")
        print(f"  ├ Avg chars/page : {char_count // max(page_count, 1):,}")
        print(f"  ├ MD5 hash       : {compute_file_hash(pdf_path)}")
        print(f"  └ Sample text    : {docs[start_idx].page_content[:120]!r}")
        print()

    print(f"  {'─' * 48}")
    print(f"  Total documents : {len(files_to_index)}")
    print(f"  Total pages     : {len(docs)}")
    print(f"  Total chars     : {sum(len(d.page_content) for d in docs):,}\n")

    # ── STEP 2: SPLIT TEXT INTO CHUNKS ────────────────────────────────────
    # Split pages into smaller overlapping chunks so the retriever can find
    # precise answers without exceeding the LLM context window.
    print("=" * 50)
    print("STEP 2: Splitting Text into Chunks")
    print("=" * 50)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,                         # Max characters per chunk
        chunk_overlap=150,                       # Overlap preserves context at chunk boundaries
        separators=["\n\n", "\n", ".", " "]      # Try to split on paragraph > line > sentence > word
    )
    splits = splitter.split_documents(docs)      # Returns a flat list of Document chunks

    for idx, chunk in enumerate(splits):
        chunk.metadata["chunk_id"] = f"{chunk.metadata['doc_id']}_chunk_{idx}"  # Unique ID for each chunk

    print(f"  Total chunks created : {len(splits)}")
    print(f"  Chunk size           : 500 chars | Overlap: 50 chars")
    print(f"  Sample chunk (index 0):\n  {splits[0].page_content[:300]!r}")
    print(f"  Metadata             : {splits[0].metadata}\n")

    chunk_dist = Counter(c.metadata["filename"] for c in splits)  # Count chunks per document
    print("  Chunk distribution per document:")
    for fname, count in chunk_dist.items():
        print(f"    {fname}: {count} chunks")
    print()

    # ── STEP 3: HUGGINGFACE EMBEDDING ─────────────────────────────────────
    # Convert text chunks into dense numerical vectors using the HuggingFace model.
    # These vectors capture semantic meaning and enable similarity search.
    print("=" * 50)
    print("STEP 3: Generating Embeddings (HuggingFace)")
    print("=" * 50)

    embeddings       = build_embeddings()
    sample_embedding = embeddings.embed_query(splits[0].page_content)  # Embed one chunk to verify model output
    print(f"  Embedding model      : {HF_MODEL}")
    print(f"  Embedding dimensions : {len(sample_embedding)}")          # Should be 384 for all-MiniLM-L6-v2
    print(f"  Device               : CPU")
    print(f"  Sample vector (first 5 values): {sample_embedding[:5]}\n")

    # ── STEP 4: INMEMORY VECTOR STORE ─────────────────────────────────────
    # Store all chunk embeddings in RAM for fast similarity search.
    # SQLRecordManager tracks which chunks are already indexed to avoid duplicates.
    print("=" * 50)
    print("STEP 4: Storing Chunks into InMemory Vector Store")
    print("=" * 50)

    vectorstore    = build_vectorstore(embeddings)
    retriever      = vectorstore.as_retriever(search_kwargs={"k": 4})  # Return top-4 chunks per query
    record_manager = SQLRecordManager(namespace="inmemory/hf_docs", db_url=RECORD_DB)  # Dedup tracker
    record_manager.create_schema()  # Create the SQLite tracking table if it doesn't exist

    for pdf_path, doc_id in files_to_index:
        filename   = Path(pdf_path).name
        doc_splits = [c for c in splits if c.metadata["doc_id"] == doc_id]  # Only chunks from this PDF
        result     = index(
            doc_splits,
            record_manager,
            vectorstore,
            cleanup="incremental",      # Only add new/changed chunks, delete removed ones
            source_id_key="doc_id",     # Group chunks by document for cleanup tracking
        )
        print(f"  Index attempt [{filename}]: {result}")  # Shows num_added, num_skipped, etc.

    print(f"\n  Vector store         : InMemoryVectorStore")
    print(f"  Embedding model      : HuggingFace ({HF_MODEL})")
    print(f"  Total chunks indexed : {len(splits)}")
    print(f"  Retriever top-k      : 4\n")

    for pdf_path, doc_id in files_to_index:
        doc_chunks = sum(1 for c in splits if c.metadata["doc_id"] == doc_id)  # Count chunks for this PDF
        update_tracker(pdf_path, tracker, doc_chunks, doc_id)  # Save hash + chunk count to tracker
    print(f"  Tracker updated      : {TRACKER_FILE}\n")

    # ── STEP 4b: PGVECTOR STORAGE ─────────────────────────────────────────
    # Store vectors persistently in PostgreSQL using the PGVector extension.
    # Skipped gracefully if PostgreSQL is not running.
    print("=" * 50)
    print("STEP 4b: Storing Embeddings into PGVector")
    print("=" * 50)

    pg_vectorstore = build_pg_vectorstore(splits, embeddings)  # Returns None if DB unavailable
    if pg_vectorstore:
        pg_retriever      = pg_vectorstore.as_retriever(search_kwargs={"k": 4})
        pg_record_manager = SQLRecordManager(namespace="pgvector/hf_docs", db_url=RECORD_DB)  # Separate namespace from InMemory
        pg_record_manager.create_schema()
        print(f"  Connection           : {PG_CONNECTION}")
        print(f"  Collection           : {COLLECTION_NAME}")
        print(f"  Embedding model      : HuggingFace ({HF_MODEL})")
        print(f"  Total vectors stored : {len(splits)}")
        for pdf_path, doc_id in files_to_index:
            filename   = Path(pdf_path).name
            doc_splits = [c for c in splits if c.metadata["doc_id"] == doc_id]
            result     = index(
                doc_splits,
                pg_record_manager,
                pg_vectorstore,
                cleanup="incremental",
                source_id_key="doc_id",
            )
            print(f"  PGVector index [{filename}]: {result}")
        pg_results = pg_vectorstore.similarity_search_with_score("how does attention work", k=3)  # Quick test query
        print(f"\n  PGVector Similarity Search: 'how does attention work'")
        for doc, score in pg_results:
            print(f"    Score: {score:.4f} | {doc.metadata.get('filename','?')} | Page {doc.metadata.get('page','?')} | {doc.page_content[:100]!r}")
        print()
    else:
        print(f"  To enable: start PostgreSQL with Docker and update PG_CONNECTION.\n")
        pg_retriever = None  # Fall back to InMemory retriever in subsequent steps

    # ── STEP 4c: VECTOR STORE OPERATIONS ──────────────────────────────────
    # Demonstrate add, similarity search, MMR search, and delete operations.
    print("=" * 50)
    print("STEP 4c: Vector Store Operations")
    print("=" * 50)

    # ADD: manually insert extra documents into the vector store
    new_docs = [
        Document(page_content="The Transformer uses self-attention to compute representations of input and output.", metadata={"source": "manual", "doc_id": "manual", "page": 0}),
        Document(page_content="Multi-head attention allows the model to jointly attend to information from different subspaces.", metadata={"source": "manual", "doc_id": "manual", "page": 0}),
    ]
    added_ids = vectorstore.add_documents(new_docs)  # Returns list of assigned vector IDs
    print(f"  ADD    : Added {len(added_ids)} documents | IDs: {added_ids}")

    # SIMILARITY SEARCH: returns top-k chunks ranked by cosine similarity
    sim_query   = "how does attention mechanism work"
    sim_results = vectorstore.similarity_search_with_score(sim_query, k=3)
    print(f"\n  SIMILARITY SEARCH: '{sim_query}'")
    for doc, score in sim_results:
        print(f"    Score: {score:.4f} | Page {doc.metadata.get('page','?')} | {doc.page_content[:100]!r}")

    # MMR SEARCH: returns diverse results by penalizing near-duplicate chunks
    mmr_results = vectorstore.max_marginal_relevance_search(sim_query, k=3, fetch_k=10)  # fetch_k=10 candidates, return 3 diverse ones
    print(f"\n  MMR SEARCH (diverse): '{sim_query}'")
    for i, doc in enumerate(mmr_results):
        print(f"    [{i+1}] Page {doc.metadata.get('page','?')} | {doc.page_content[:100]!r}")

    # DELETE: remove the manually added documents by their IDs
    vectorstore.delete(added_ids)
    print(f"\n  DELETE : Removed {len(added_ids)} manually added documents\n")

    # ── STEP 5: SEMANTIC SEARCH ────────────────────────────────────────────
    # Run a test query to verify the retriever returns relevant chunks.
    print("=" * 50)
    print("STEP 5: Semantic Search (Retrieval Test)")
    print("=" * 50)

    test_query = "What is the Transformer architecture?"
    retrieved  = retriever.invoke(test_query)  # Returns top-k most relevant chunks
    print(f"  Query: '{test_query}'")
    print(f"  Top {len(retrieved)} retrieved chunks:\n")
    for i, doc in enumerate(retrieved):
        print(f"  [{i+1}] {doc.metadata.get('filename','?')} | Page {doc.metadata.get('page','?')} | {doc.page_content[:150]!r}\n")

    # ── STEP 6: RAG GENERATION ─────────────────────────────────────────────
    # Build the full RAG chain: retriever → prompt → LLM → output parser.
    print("=" * 50)
    print("STEP 6: RAG Generation (HuggingFace Embeddings + Ollama LLM)")
    print("=" * 50)

    llm       = ChatOllama(model="llama3", temperature=0)  # temperature=0 = deterministic responses
    rag_chain = build_rag_chain(retriever, llm)             # Assemble the full RAG pipeline

    # ── STEP 7: INTERACTIVE Q&A ────────────────────────────────────────────
    # Accept user questions, retrieve relevant chunks, and generate answers.
    print("\n" + "=" * 50)
    print("RAG Pipeline Ready! Type 'exit' to quit.")
    print("Embeddings: HuggingFace | LLM: Ollama llama3")
    print("=" * 50 + "\n")

    while True:
        question = input("Ask about the indexed documents: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Exiting RAG pipeline.")
            break
        if not question:
            continue                                        # Skip empty input
        print("\nRetrieving and generating answer...")
        answer = rag_chain.invoke(question)                 # Retrieve chunks + generate answer
        print(f"\nAnswer: {answer}\n")
        print("-" * 50 + "\n")
