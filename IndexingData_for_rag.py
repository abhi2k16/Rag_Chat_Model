import os
from pathlib import Path

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

file_name = "Attention_is_All_You_Need.pdf"
PDF_PATH = os.path.join(PDF_FOLDER, file_name)
# ─────────────────────────────────────────────
# STEP 1: LOAD DOCUMENT
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading Document")
print("=" * 50)

docs = []
with pdfplumber.open(PDF_PATH) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text and text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": PDF_PATH, "page": i + 1}
            ))

print(f"  Loaded {len(docs)} pages from: {PDF_PATH}")
print(f"  Sample (Page 1, first 200 chars):\n  {docs[0].page_content[:200]!r}\n")

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

print(f"  Total chunks created : {len(splits)}")
print(f"  Chunk size           : 500 chars | Overlap: 50 chars")
print(f"  Sample chunk (index 0):\n  {splits[0].page_content[:300]!r}")
print(f"  Metadata             : {splits[0].metadata}\n")

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
# STEP 4: STORE INTO VECTOR STORE
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 4: Storing Chunks into Vector Store")
print("=" * 50)

print(f"  Embedding and indexing {len(splits)} chunks... (this may take a moment)")
vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

print(f"  Vector store         : InMemoryVectorStore")
print(f"  Total vectors stored : {len(splits)}")
print(f"  Retriever top-k      : 4\n")

# ─────────────────────────────────────────────
# STEP 4b: STORE INTO PGVECTOR
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 4b: Storing Chunks into PGVector")
print("=" * 50)

# PostgreSQL connection string — update credentials as needed
user = "Langchain"  # <-- CHANGE THIS to your PostgreSQL username
password = "Langchain"  # <-- CHANGE THIS to your PostgreSQL password
PG_CONNECTION = f"postgresql+psycopg://{user}:{password}@localhost:6024/Langchain"
COLLECTION_NAME = "attention_paper"

try:
    pg_vectorstore = PGVector.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=PG_CONNECTION,
        use_jsonb=True,
        pre_delete_collection=True,   # fresh index on every run
    )
    pg_retriever = pg_vectorstore.as_retriever(search_kwargs={"k": 4})
    print(f"  Connection           : {PG_CONNECTION}")
    print(f"  Collection           : {COLLECTION_NAME}")
    print(f"  Total vectors stored : {len(splits)}")

    # Similarity search with score
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
        f"[Page {d.metadata.get('page', '?')}]: {d.page_content}" for d in docs
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
    question = input("Ask about 'Attention Is All You Need' paper: ").strip()
    if question.lower() in ("exit", "quit"):
        print("Exiting RAG pipeline.")
        break
    if not question:
        continue
    print("\nRetrieving and generating answer...")
    answer = rag_chain.invoke(question)
    print(f"\nAnswer: {answer}\n")
    print("-" * 50 + "\n")


