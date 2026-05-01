"""
retrieval_rag.py
─────────────────────────────────────────────
Retrieval stage of the RAG pipeline.

Responsibilities:
  - Load PDF documents from disk and extract text
  - Split text into overlapping chunks for better context coverage
  - Build HuggingFace embeddings and store vectors in InMemory / PGVector
  - Expose reusable search functions (similarity, MMR, filtered)
  - Can be run standalone for retrieval testing

Imports reusable config and builder functions from:
  IndexingData_for_rag_Ollama_LangChain_HuggingFace.py

Exported functions used by generation_rag_Ollama_LangChain_HuggingFace.py:
  - load_and_split_docs()
  - build_stores()
"""

import os
import sys
from pathlib import Path

# ── Fix broken SSL_CERT_FILE env variable ──────────────────────────────────
# Some Anaconda environments set SSL_CERT_FILE to a path that no longer exists.
# This causes httpx (used by Ollama) to fail on startup. We remove it if invalid.
ssl_cert_file = os.environ.get("SSL_CERT_FILE")
if ssl_cert_file and not Path(ssl_cert_file).is_file():
    os.environ.pop("SSL_CERT_FILE", None)

# ── Add project folder to sys.path so sibling modules can be imported ──────
sys.path.insert(0, str(Path(__file__).parent))

# ── Import config constants and builder functions from the indexing module ──
from IndexingDocs_for_rag import (
    # File paths and DB connection settings
    PDF_FILES, PDF_FOLDER, TRACKER_FILE, RECORD_DB,
    # Embedding model name and PGVector connection details
    HF_MODEL, PG_CONNECTION, COLLECTION_NAME,
    # Utility functions for text cleaning and document tracking
    clean_text, compute_file_hash,
    load_tracker, get_or_create_doc_id,
    # Builder functions for embeddings, vector stores, retriever, RAG chain
    build_embeddings, build_vectorstore, build_pg_vectorstore,
    build_rag_chain, build_retriever, build_prompt, format_docs,
)

import pdfplumber                                          # PDF text extraction
from tqdm import tqdm                                      # Progress bars
from langchain_core.documents import Document              # LangChain document wrapper
from langchain_core.vectorstores import InMemoryVectorStore  # In-memory vector store
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Text chunking
from langchain_classic.indexes import index, SQLRecordManager  # Incremental indexing
from langchain_postgres.vectorstores import PGVector       # PostgreSQL vector store
from langchain_ollama import ChatOllama                    # Ollama LLM wrapper


# ═══════════════════════════════════════════════════════════════════════════
# REUSABLE FUNCTIONS  (imported by generation_rag_Ollama_LangChain_HuggingFace.py)
# ═══════════════════════════════════════════════════════════════════════════

def load_and_split_docs():
    """
    Load all PDF files listed in PDF_FILES, extract and clean text page by page,
    then split the text into overlapping chunks for embedding.

    Steps:
      1. Load the document change tracker to get/create stable doc_ids
      2. Open each PDF with pdfplumber and extract text per page
      3. Clean each page (remove garbled chars, collapse whitespace)
      4. Skip pages with less than 30 chars (likely figure-only pages)
      5. Split all pages into 500-char chunks with 50-char overlap
      6. Assign a unique chunk_id to each chunk for traceability

    Returns:
      splits         : list of LangChain Document chunks with metadata
      files_to_index : list of (pdf_path, doc_id) tuples
    """
    # Load the tracker JSON that stores doc_ids and file hashes across runs
    tracker        = load_tracker()
    # Build list of (path, doc_id) — reuses existing doc_id if file was seen before
    files_to_index = [(p, get_or_create_doc_id(p, tracker)) for p in PDF_FILES]
    docs = []  # Will hold one Document per valid page

    # ── STAGE 1: Document Loading ───────────────────────────────────────────
    for pdf_path, doc_id in files_to_index:
        filename   = Path(pdf_path).name
        page_count = 0  # Count of pages that passed the content filter

        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages
            # tqdm shows a per-page progress bar for each document
            with tqdm(total=len(pages), desc=f"    Loading '{filename}'", unit="page", ncols=70) as pbar:
                for i, page in enumerate(pages):
                    text = page.extract_text()  # Extract raw text from the PDF page

                    if text and text.strip():
                        text = clean_text(text)  # Remove garbled chars and normalize whitespace

                        # Skip pages that are mostly empty after cleaning (e.g. figure pages)
                        if len(text) >= 30:
                            docs.append(Document(
                                page_content=text,
                                metadata={
                                    "source"  : pdf_path,   # Full file path
                                    "filename": filename,   # Just the filename
                                    "doc_id"  : doc_id,     # Stable UUID for this document
                                    "page"    : i + 1,      # 1-based page number
                                }
                            ))
                            page_count += 1
                    pbar.update(1)  # Advance the progress bar by one page

        print(f"    ✔ Loaded : {filename} | {page_count} pages | doc_id: {doc_id}")
    print(f"    ✔ Total pages loaded : {len(docs)}")

    # ── STAGE 2: Text Splitting ─────────────────────────────────────────────
    # RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    # then newlines, then sentences, then spaces — preserving semantic boundaries.
    print("    Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,                          # Max characters per chunk
        chunk_overlap=50,                        # Overlap to preserve context at boundaries
        separators=["\n\n", "\n", ".", " "]      # Split priority order
    )

    # Split page by page and show progress
    with tqdm(total=len(docs), desc="    Splitting pages", unit="page", ncols=70) as pbar:
        splits = []
        for doc in docs:
            splits.extend(splitter.split_documents([doc]))  # Split one page at a time
            pbar.update(1)

    # ── STAGE 3: Assign Chunk IDs ───────────────────────────────────────────
    # Each chunk gets a unique ID combining its doc_id and position index.
    # This allows tracing any chunk back to its source document.
    with tqdm(total=len(splits), desc="    Assigning chunk IDs", unit="chunk", ncols=70) as pbar:
        for idx, chunk in enumerate(splits):
            chunk.metadata["chunk_id"] = f"{chunk.metadata['doc_id']}_chunk_{idx}"
            pbar.update(1)

    print(f"    ✔ Text splitting complete : {len(splits)} chunks (size=500, overlap=50)")
    return splits, files_to_index


def build_stores(splits, files_to_index):
    """
    Build the embedding model and vector stores from the provided chunks.

    Steps:
      1. Initialize HuggingFace sentence-transformer embedding model
      2. Create an InMemoryVectorStore and index all chunks incrementally
         using SQLRecordManager to avoid re-indexing unchanged chunks
      3. Attempt to connect to PGVector (PostgreSQL) for persistent storage

    Args:
      splits         : list of Document chunks from load_and_split_docs()
      files_to_index : list of (pdf_path, doc_id) tuples

    Returns:
      embeddings     : HuggingFaceEmbeddings instance
      vectorstore    : InMemoryVectorStore with all chunks indexed
      pg_vectorstore : PGVector instance or None if unavailable
    """

    # ── STAGE 1: Initialize Embedding Model ────────────────────────────────
    # HuggingFace all-MiniLM-L6-v2 runs fully locally — no API key needed.
    # It produces 384-dimensional normalized vectors.
    print("    Initializing HuggingFace embeddings...")
    embeddings = build_embeddings()
    print(f"    ✔ Embeddings ready : {HF_MODEL}")

    # Quick test to confirm the model is working and show vector dimensions
    with tqdm(total=1, desc="    Testing embedding", unit="sample", ncols=70) as pbar:
        dim = len(embeddings.embed_query("test"))
        pbar.update(1)
    print(f"    ✔ Embedding dimensions : {dim}")

    # ── STAGE 2: InMemory Vector Store ─────────────────────────────────────
    # InMemoryVectorStore holds all vectors in RAM for fast retrieval.
    # SQLRecordManager tracks which chunks have already been indexed so
    # re-running the pipeline skips unchanged chunks (incremental indexing).
    print("    Indexing chunks into InMemoryVectorStore...")
    vectorstore    = build_vectorstore(embeddings)
    record_manager = SQLRecordManager(namespace="retrieval/hf_docs", db_url=RECORD_DB)
    record_manager.create_schema()  # Creates the SQLite tracking table if it doesn't exist

    for pdf_path, doc_id in files_to_index:
        filename   = Path(pdf_path).name
        # Filter only the chunks belonging to this specific document
        doc_splits = [c for c in splits if c.metadata["doc_id"] == doc_id]

        # Index in batches of 10 so the progress bar updates smoothly
        batch_size = 10
        with tqdm(total=len(doc_splits), desc=f"    Embedding '{filename}'", unit="chunk", ncols=70) as pbar:
            for i in range(0, len(doc_splits), batch_size):
                batch = doc_splits[i:i + batch_size]
                # cleanup="incremental" — only adds new/changed chunks, deletes removed ones
                # source_id_key="doc_id" — groups chunks by document for cleanup tracking
                index(batch, record_manager, vectorstore, cleanup="incremental", source_id_key="doc_id")
                pbar.update(len(batch))
        print(f"    ✔ InMemory indexed : {filename} ({len(doc_splits)} chunks)")
    print(f"    ✔ InMemoryVectorStore ready : {len(splits)} total vectors")

    # ── STAGE 3: PGVector (PostgreSQL) ─────────────────────────────────────
    # PGVector stores vectors persistently in PostgreSQL.
    # If the DB is not running, this step is gracefully skipped.
    print("    Connecting to PGVector...")
    pg_vectorstore = build_pg_vectorstore(splits, embeddings)
    if pg_vectorstore:
        print(f"    ✔ PGVector ready : {len(splits)} vectors stored")
    else:
        print(f"    ⚠ PGVector skipped : not available")
    return embeddings, vectorstore, pg_vectorstore


# ═══════════════════════════════════════════════════════════════════════════
# SEARCH FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_similarity_search(query: str, store, k: int = 4, label: str = "InMemory"):
    """
    Run a standard cosine similarity search against the vector store.
    Returns the top-k most similar chunks along with their similarity scores.

    Args:
      query : the search query string
      store : InMemoryVectorStore or PGVector instance
      k     : number of results to return
      label : display label for the output header
    """
    results = store.similarity_search_with_score(query, k=k)
    print(f"\n  [{label}] Similarity Search: '{query}'")
    for doc, score in results:
        print(f"    Score: {score:.4f} | {doc.metadata.get('filename','?')} | Page {doc.metadata.get('page','?')} | {doc.page_content[:120]!r}")
    return results


def run_mmr_search(query: str, store, k: int = 4, fetch_k: int = 10, label: str = "InMemory"):
    """
    Run a Maximal Marginal Relevance (MMR) search.
    MMR balances relevance and diversity — it avoids returning near-duplicate chunks
    by penalizing results that are too similar to already-selected ones.

    Args:
      query   : the search query string
      store   : InMemoryVectorStore or PGVector instance
      k       : number of final results to return
      fetch_k : number of candidates to fetch before MMR re-ranking
      label   : display label for the output header
    """
    results = store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    print(f"\n  [{label}] MMR Search (diverse): '{query}'")
    for i, doc in enumerate(results):
        print(f"    [{i+1}] {doc.metadata.get('filename','?')} | Page {doc.metadata.get('page','?')} | {doc.page_content[:120]!r}")
    return results


def run_filtered_search(query: str, store, filter_filename: str, k: int = 4):
    """
    Run a similarity search restricted to a specific document by filename.

    InMemoryVectorStore does not support native dict-based filters,
    so we fetch a larger result set and post-filter in Python.
    PGVector supports native metadata filtering via the filter parameter.

    Args:
      query           : the search query string
      store           : InMemoryVectorStore or PGVector instance
      filter_filename : only return chunks from this filename
      k               : number of results to return
    """
    if isinstance(store, InMemoryVectorStore):
        # Fetch k*5 results then filter by filename in Python
        all_results = store.similarity_search_with_score(query, k=k * 5)
        results = [(doc, score) for doc, score in all_results
                   if doc.metadata.get("filename") == filter_filename][:k]
    else:
        # PGVector supports native metadata filtering
        results = store.similarity_search_with_score(
            query, k=k,
            filter={"filename": filter_filename}
        )
    print(f"\n  [Filtered] Search in '{filter_filename}': '{query}'")
    for doc, score in results:
        print(f"    Score: {score:.4f} | Page {doc.metadata.get('page','?')} | {doc.page_content[:120]!r}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN RETRIEVAL PIPELINE  (runs only when executed directly)
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("=" * 50)
    print("RAG RETRIEVAL PIPELINE")
    print("Embeddings : HuggingFace")
    print("LLM        : Ollama llama3")
    print("=" * 50 + "\n")

    # ── Setup: load docs, build stores and retriever ────────────────────────
    print("Loading documents and building vector stores...")
    splits, files_to_index                  = load_and_split_docs()
    embeddings, vectorstore, pg_vectorstore = build_stores(splits, files_to_index)

    # Use PGVector retriever if available, otherwise fall back to InMemory
    retriever, active_store = build_retriever(vectorstore, pg_vectorstore, k=4)
    pg_retriever = pg_vectorstore.as_retriever(search_kwargs={"k": 4}) if pg_vectorstore else None

    print(f"  Total chunks : {len(splits)}")
    print(f"  Active store : {active_store}\n")

    # ── STEP 3: Retrieval Tests ─────────────────────────────────────────────
    print("=" * 50)
    print("STEP 3: Retrieval — Similarity & MMR Search")
    print("=" * 50)

    test_query = "What is the Transformer architecture?"

    # Test similarity search on InMemory store
    run_similarity_search(test_query, vectorstore, label="InMemory")
    run_mmr_search(test_query, vectorstore, label="InMemory")

    # Also test on PGVector if it's available
    if pg_vectorstore:
        run_similarity_search(test_query, pg_vectorstore, label="PGVector")
        run_mmr_search(test_query, pg_vectorstore, label="PGVector")

    # Test filtered search — only search within the Attention paper
    run_filtered_search("multi-head attention", vectorstore, filter_filename="Attention_is_All_You_Need.pdf")
    print()

    # ── STEP 4: RAG Generation ──────────────────────────────────────────────
    print("=" * 50)
    print("STEP 4: RAG Generation (HuggingFace + Ollama)")
    print("=" * 50)

    llm              = ChatOllama(model="llama3", temperature=0)
    active_retriever = pg_retriever if pg_retriever else retriever  # Prefer PGVector
    print(f"  Active vector store  : {active_store}")
    print(f"  Embedding model      : {HF_MODEL}")
    print(f"  LLM                  : Ollama llama3\n")
    rag_chain = build_rag_chain(active_retriever, llm)

    # ── STEP 5: Interactive Q&A ─────────────────────────────────────────────
    print("=" * 50)
    print("RAG Retrieval Ready! Type 'exit' to quit.")
    print("Commands:")
    print("  sim:<query>  — run similarity search only")
    print("  mmr:<query>  — run MMR search only")
    print("  <query>      — full RAG answer")
    print("=" * 50 + "\n")

    while True:
        user_input = input("Query: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Exiting retrieval pipeline.")
            break
        if not user_input:
            continue

        if user_input.startswith("sim:"):
            # Run similarity search only — no LLM generation
            query = user_input[4:].strip()
            run_similarity_search(query, vectorstore, label="InMemory")
            if pg_vectorstore:
                run_similarity_search(query, pg_vectorstore, label="PGVector")

        elif user_input.startswith("mmr:"):
            # Run MMR search only — no LLM generation
            query = user_input[4:].strip()
            run_mmr_search(query, vectorstore, label="InMemory")
            if pg_vectorstore:
                run_mmr_search(query, pg_vectorstore, label="PGVector")

        else:
            # Full RAG: retrieve relevant chunks then generate answer with LLM
            print("\nRetrieving and generating answer...")
            answer = rag_chain.invoke(user_input)
            print(f"\nAnswer: {answer}\n")
            print("-" * 50 + "\n")
