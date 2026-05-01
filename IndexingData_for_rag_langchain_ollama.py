import os
import json
import hashlib
import uuid
from pathlib import Path
from datetime import datetime
from collections import Counter

# Fix SSL_CERT_FILE if it points to a missing file
ssl_cert_file = os.environ.get("SSL_CERT_FILE")
if ssl_cert_file and not Path(ssl_cert_file).is_file():
    os.environ.pop("SSL_CERT_FILE", None)

import pdfplumber
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

PDF_FOLDER = "c:\\Users\\abhij\\Desktop\\GenAIwithLLMs\\LangChain_projects\\"
TRACKER_FILE = os.path.join(PDF_FOLDER, "doc_change_tracker.json")

# Add all PDF files to index here
PDF_FILES = [
    os.path.join(PDF_FOLDER, "Attention_is_All_You_Need.pdf"),
    os.path.join(PDF_FOLDER, "MachineLearning-Lecture01.pdf"),
]

# ─────────────────────────────────────────────
# DOCUMENT CHANGE TRACKING UTILITIES
# ─────────────────────────────────────────────

def compute_file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file to detect content changes."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_tracker() -> dict:
    """Load the change tracker JSON from disk."""
    if Path(TRACKER_FILE).exists():
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_tracker(tracker: dict):
    """Save the change tracker JSON to disk."""
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)

def check_document_changes(filepath: str, tracker: dict) -> tuple[bool, str]:
    """
    Compare current file hash against stored hash.
    Returns (has_changed, status_message).
    """
    current_hash = compute_file_hash(filepath)
    filename = Path(filepath).name

    if filename not in tracker:
        return True, "NEW"
    elif tracker[filename]["hash"] != current_hash:
        return True, "MODIFIED"
    else:
        return False, "UNCHANGED"

def update_tracker(filepath: str, tracker: dict, num_chunks: int, doc_id: str):
    """Update tracker with latest file hash, timestamp, chunk count and doc_id."""
    filename = Path(filepath).name
    current_hash = compute_file_hash(filepath)
    now = datetime.now().isoformat()

    previous = tracker.get(filename, {})
    tracker[filename] = {
        "doc_id"          : doc_id,
        "hash"            : current_hash,
        "last_indexed"    : now,
        "num_chunks"      : num_chunks,
        "previous_hash"   : previous.get("hash", None),
        "previous_indexed": previous.get("last_indexed", None),
    }
    save_tracker(tracker)

def get_or_create_doc_id(filepath: str, tracker: dict) -> str:
    """Reuse existing doc_id if file was seen before, else generate a new one."""
    filename = Path(filepath).name
    if filename in tracker and "doc_id" in tracker[filename]:
        return tracker[filename]["doc_id"]
    return str(uuid.uuid4())

# ─────────────────────────────────────────────
# STEP 0: DOCUMENT CHANGE TRACKING
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 0: Document Change Tracking")
print("=" * 50)

tracker = load_tracker()
files_to_index = []

for pdf_path in PDF_FILES:
    filename = Path(pdf_path).name
    has_changed, status = check_document_changes(pdf_path, tracker)
    doc_id = get_or_create_doc_id(pdf_path, tracker)
    print(f"  File     : {filename}")
    print(f"  Doc ID   : {doc_id}")
    print(f"  Status   : {status}")
    if status == "NEW":
        print(f"  Action   : First time indexing.")
        files_to_index.append((pdf_path, doc_id))
    elif status == "MODIFIED":
        prev = tracker[filename]
        print(f"  Prev indexed : {prev['last_indexed']} | Prev hash: {prev['hash']}")
        print(f"  Action   : Document changed — re-indexing.")
        files_to_index.append((pdf_path, doc_id))
    else:
        prev = tracker[filename]
        print(f"  Last indexed : {prev['last_indexed']} | Chunks: {prev['num_chunks']}")
        print(f"  Action   : No changes — loading from tracker.")
        files_to_index.append((pdf_path, doc_id))  # still load for RAG
    print()

# ─────────────────────────────────────────────
# STEP 1: LOAD MULTIPLE DOCUMENTS
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading Multiple Documents")
print("=" * 50)

docs = []
for pdf_path, doc_id in files_to_index:
    filename = Path(pdf_path).name
    page_count = 0
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source"  : pdf_path,
                        "filename": filename,
                        "doc_id"  : doc_id,
                        "page"    : i + 1,
                    }
                ))
                page_count += 1
    print(f"  Loaded {page_count} pages | doc_id: {doc_id} | file: {filename}")

print(f"\n  Total pages loaded : {len(docs)}")
print(f"  Sample metadata    : {docs[0].metadata}\n")

# ─────────────────────────────────────────────
# STEP 2: SPLIT TEXT INTO CHUNKS
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 2: Splitting Text into Chunks")
print("=" * 50)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
splits = splitter.split_documents(docs)

# Assign unique chunk_id to each split
for idx, chunk in enumerate(splits):
    chunk.metadata["chunk_id"] = f"{chunk.metadata['doc_id']}_chunk_{idx}"

print(f"  Total chunks created : {len(splits)}")
print(f"  Chunk size           : 500 chars | Overlap: 50 chars")
print(f"  Sample chunk (index 0):\n  {splits[0].page_content[:300]!r}")
print(f"  Metadata             : {splits[0].metadata}\n")

# Show chunk distribution per document
chunk_dist = Counter(c.metadata["filename"] for c in splits)
print("  Chunk distribution per document:")
for fname, count in chunk_dist.items():
    print(f"    {fname}: {count} chunks")
print()

# ─────────────────────────────────────────────
# STEP 3: EMBEDDING
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 3: Generating Embeddings")
print("=" * 50)

embeddings = OllamaEmbeddings(model="llama3")

sample_embedding = embeddings.embed_query(splits[0].page_content)
print(f"  Embedding model      : llama3 (via Ollama)")
print(f"  Embedding dimensions : {len(sample_embedding)}")
print(f"  Sample vector (first 5 values): {sample_embedding[:5]}\n")

# ─────────────────────────────────────────────
# STEP 4: STORE INTO VECTOR STORE + INDEX API
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 4: Storing Chunks into Vector Store")
print("=" * 50)

from langchain_classic.indexes import index, SQLRecordManager

# InMemoryVectorStore
vectorstore = InMemoryVectorStore(embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# SQLite-backed RecordManager to track what has been indexed
RECORD_DB = f"sqlite:///{PDF_FOLDER}record_manager.db"
record_manager = SQLRecordManager(namespace="inmemory/docs", db_url=RECORD_DB)
record_manager.create_schema()

# Index each document separately so we can report per-doc results
for pdf_path, doc_id in files_to_index:
    filename = Path(pdf_path).name
    doc_splits = [c for c in splits if c.metadata["doc_id"] == doc_id]
    result = index(
        doc_splits,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="doc_id",
    )
    print(f"  Index attempt [{filename}]: {result}")

print(f"\n  Vector store         : InMemoryVectorStore")
print(f"  Total vectors stored : {len(splits)}")
print(f"  Retriever top-k      : 4\n")

# Update tracker for all indexed documents
for pdf_path, doc_id in files_to_index:
    doc_chunks = sum(1 for c in splits if c.metadata["doc_id"] == doc_id)
    update_tracker(pdf_path, tracker, doc_chunks, doc_id)
print(f"  Tracker updated      : {TRACKER_FILE}\n")

# ─────────────────────────────────────────────
# STEP 4b: STORE INTO PGVECTOR
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 4b: Storing Chunks into PGVector")
print("=" * 50)

# PostgreSQL connection string — update credentials as needed
user = "Langchain"
password = "Langchain"
PG_CONNECTION = f"postgresql+psycopg://{user}:{password}@localhost:6024/Langchain"
COLLECTION_NAME = "attention_paper"

try:
    pg_vectorstore = PGVector.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=PG_CONNECTION,
        use_jsonb=True,
        pre_delete_collection=True,
    )
    pg_retriever = pg_vectorstore.as_retriever(search_kwargs={"k": 4})
    print(f"  Connection           : {PG_CONNECTION}")
    print(f"  Collection           : {COLLECTION_NAME}")
    print(f"  Total vectors stored : {len(splits)}")

    pg_results = pg_vectorstore.similarity_search_with_score("how does attention work", k=3)
    print(f"\n  PGVector Similarity Search: 'how does attention work'")
    for doc, score in pg_results:
        print(f"    Score: {score:.4f} | Page {doc.metadata.get('page','?')} | {doc.page_content[:100]!r}")
    print()
except Exception as e:
    print(f"  [SKIPPED] PGVector not available: {e}")
    print(f"  To enable: start PostgreSQL, run 'CREATE EXTENSION vector;' and update PG_CONNECTION.\n")
    pg_retriever = None

# ─────────────────────────────────────────────
# STEP 4c: VECTOR STORE OPERATIONS (InMemory)
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 4c: Vector Store Operations (InMemory)")
print("=" * 50)

# --- ADD new documents ---
new_docs = [
    Document(page_content="The Transformer uses self-attention to compute representations of input and output.", metadata={"source": "manual", "page": 0}),
    Document(page_content="Multi-head attention allows the model to jointly attend to information from different representation subspaces.", metadata={"source": "manual", "page": 0}),
]
added_ids = vectorstore.add_documents(new_docs)
print(f"  ADD   : Added {len(added_ids)} new documents | IDs: {added_ids}")

# --- SIMILARITY SEARCH with scores ---
sim_query = "how does attention mechanism work"
sim_results = vectorstore.similarity_search_with_score(sim_query, k=3)
print(f"\n  SIMILARITY SEARCH: '{sim_query}'")
for doc, score in sim_results:
    print(f"    Score: {score:.4f} | Page {doc.metadata.get('page','?')} | {doc.page_content[:100]!r}")

# --- MMR SEARCH (Maximal Marginal Relevance) for diverse results ---
mmr_results = vectorstore.max_marginal_relevance_search(sim_query, k=3, fetch_k=10)
print(f"\n  MMR SEARCH (diverse results): '{sim_query}'")
for i, doc in enumerate(mmr_results):
    print(f"    [{i+1}] Page {doc.metadata.get('page','?')} | {doc.page_content[:100]!r}")

# --- DELETE documents by ID ---
vectorstore.delete(added_ids)
print(f"\n  DELETE: Removed {len(added_ids)} manually added documents | IDs: {added_ids}\n")

# ─────────────────────────────────────────────
# STEP 5: SEMANTIC SEARCH
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 5: Semantic Search (Retrieval Test)")
print("=" * 50)

test_query = "What is the Transformer architecture?"
retrieved = retriever.invoke(test_query)

print(f"  Query: '{test_query}'")
print(f"  Top {len(retrieved)} retrieved chunks:\n")
for i, doc in enumerate(retrieved):
    print(f"  [{i+1}] Page {doc.metadata.get('page', '?')} | {doc.page_content[:150]!r}\n")

# ─────────────────────────────────────────────
# STEP 6: RAG GENERATION
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 6: RAG Generation with Ollama LLM")
print("=" * 50)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}
""")

llm = ChatOllama(model="llama3", temperature=0)

def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('filename','?')} | Page {d.metadata.get('page','?')} | doc_id: {d.metadata.get('doc_id','?')}]: {d.page_content}"
        for d in docs
    )

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ─────────────────────────────────────────────
# STEP 7: INTERACTIVE Q&A
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("RAG Pipeline Ready! Type 'exit' to quit.")
print("=" * 50 + "\n")

while True:
    question = input("Ask about the indexed documents: ").strip()
    if question.lower() in ("exit", "quit"):
        print("Exiting RAG pipeline.")
        break
    if not question:
        continue
    print("\nRetrieving and generating answer...")
    answer = rag_chain.invoke(question)
    print(f"\nAnswer: {answer}\n")
    print("-" * 50 + "\n")
