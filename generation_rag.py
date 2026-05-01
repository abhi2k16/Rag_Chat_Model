"""
generation_rag_Ollama_LangChain_HuggingFace.py
───────────────────────────────────────────────
Generation stage of the RAG pipeline.

Responsibilities:
  - Build and switch between multiple prompt templates
  - Support three generation modes: standard invoke, streaming, batch
  - Preview the fully rendered prompt before sending to the LLM
  - Debug retrieved chunks for a given query
  - Interactive Q&A loop with command-based controls

Imports from:
  IndexingDocs_for_rag.py  — config, builders
  retrieval_rag.py          — load_and_split_docs, build_stores
"""

import os
import sys
from pathlib import Path

# ── Fix broken SSL_CERT_FILE env variable ──────────────────────────────────
# Prevents httpx (used by Ollama) from failing due to a missing cert file
ssl_cert_file = os.environ.get("SSL_CERT_FILE")
if ssl_cert_file and not Path(ssl_cert_file).is_file():
    os.environ.pop("SSL_CERT_FILE", None)

# ── Add project folder to sys.path for sibling module imports ──────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Import config and builder functions from the indexing module ────────────
from IndexingDocs_for_rag import (
    PDF_FILES, RECORD_DB, HF_MODEL, PG_CONNECTION, COLLECTION_NAME,
    clean_text, load_tracker, get_or_create_doc_id,
    build_embeddings, build_vectorstore, build_pg_vectorstore,
    build_retriever, build_prompt, format_docs,
)

# ── Import document loading and store building from the retrieval module ────
from retrieval_rag import (
    load_and_split_docs,   # Loads PDFs, cleans text, splits into chunks
    build_stores,          # Builds embeddings + InMemory/PGVector stores
)

import pdfplumber
from pathlib import Path
from langchain_ollama import ChatOllama                    # Ollama LLM wrapper
from langchain_core.output_parsers import StrOutputParser  # Converts LLM output to string
from langchain_core.runnables import chain                 # Decorator to turn a function into a Runnable
from langchain_core.prompts import ChatPromptTemplate      # Prompt template builder


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATE REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

# Maps user-facing menu numbers to (prompt_type, description) pairs.
# prompt_type must match a key in the templates dict inside build_prompt().
PROMPT_TYPES = {
    "1": ("default",  "Standard helpful assistant"),
    "2": ("concise",  "1-2 sentence answer"),
    "3": ("detailed", "Detailed answer with citations"),
    "4": ("bullet",   "Bullet point answer"),
    "5": ("custom",   "Write your own prompt"),
}


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def build_custom_prompt() -> ChatPromptTemplate:
    """
    Interactively collect a custom prompt template from the user.
    The template must contain {context} and {question} placeholders.
    Falls back to the default prompt if placeholders are missing.
    """
    print("\n  Write your custom prompt template.")
    print("  Use {context} and {question} as placeholders.")
    print("  Enter an empty line to finish.\n")
    lines = []
    while True:
        line = input("  > ")
        if line == "":
            break
        lines.append(line)
    template = "\n".join(lines)

    # Validate that both required placeholders are present
    if "{context}" not in template or "{question}" not in template:
        print("  [WARNING] Missing {context} or {question} — using default prompt.")
        return build_prompt("default")
    return ChatPromptTemplate.from_template(template)


def select_prompt() -> tuple[ChatPromptTemplate, str]:
    """
    Display a numbered menu and let the user pick a prompt type.
    Returns the selected (ChatPromptTemplate, label) tuple.
    """
    print("\n" + "=" * 50)
    print("SELECT PROMPT TYPE")
    print("=" * 50)
    for key, (ptype, desc) in PROMPT_TYPES.items():
        print(f"  {key}. {ptype:<10} — {desc}")
    print()
    choice = input("  Enter choice [1-5] (default=1): ").strip() or "1"

    if choice == "5":
        # Let the user write their own prompt template
        prompt = build_custom_prompt()
        label  = "custom"
    else:
        # Look up the prompt type from the registry, default to "default"
        ptype, desc = PROMPT_TYPES.get(choice, PROMPT_TYPES["1"])
        prompt = build_prompt(ptype)
        label  = ptype
    print(f"\n  Selected prompt : {label}\n")
    return prompt, label


# ═══════════════════════════════════════════════════════════════════════════
# CHAIN BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def build_generation_chain(retriever, prompt: ChatPromptTemplate, llm):
    """
    Assemble the full RAG chain:
      1. Retrieve relevant chunks from the vector store using the query
      2. Format the chunks into a context string
      3. Fill the prompt template with context + question
      4. Send the filled prompt to the LLM
      5. Parse the LLM output to a plain string

    Args:
      retriever : vector store retriever (InMemory or PGVector)
      prompt    : ChatPromptTemplate with {context} and {question}
      llm       : ChatOllama instance

    Returns:
      A LangChain Runnable chain ready for invoke / stream / batch
    """
    @chain
    def prompt_chain(question: str):
        # Retrieve matching chunks, format them, then render the prompt inside the chain.
        """The @chain decorator is helpful because it turns a standard Python function into 
        a first-class LangChain object. It bridges the gap between custom Python logic and 
        the structured LangChain Expression Language (LCEL)."""
        docs = retriever.invoke(question)
        return prompt.invoke({
            "context": format_docs(docs),
            "question": question,
        })

    return prompt_chain | llm | StrOutputParser()

# ═══════════════════════════════════════════════════════════════════════════
# GENERATION MODES
# ═══════════════════════════════════════════════════════════════════════════

def generate_stream(chain, query: str):
    """
    Stream the LLM answer token by token as it is generated.
    Uses chain.stream() which yields partial tokens in real time.
    Useful for long answers where you want to see output immediately.

    Args:
      chain : the RAG chain built by build_generation_chain()
      query : the user's question string
    """
    print("\nAnswer (streaming):\n")
    for token in chain.stream(query):
        # Print each token immediately without a newline, flush to show in real time
        print(token, end="", flush=True)
    print("\n")  # Final newline after streaming completes


def generate_batch(chain, queries: list[str]) -> list[str]:
    """
    Run multiple queries in a single batch call.
    Uses chain.batch() which processes all queries in parallel internally.
    More efficient than calling invoke() multiple times in a loop.

    Args:
      chain   : the RAG chain built by build_generation_chain()
      queries : list of question strings to answer

    Returns:
      list of answer strings in the same order as queries
    """
    print(f"\n  Running batch generation for {len(queries)} queries...")
    answers = chain.batch(queries)  # All queries processed together
    for i, (q, a) in enumerate(zip(queries, answers)):
        print(f"\n  [{i+1}] Query  : {q}")
        print(f"       Answer : {a}")
    print()
    return answers


# ═══════════════════════════════════════════════════════════════════════════
# DEBUG / PREVIEW UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def show_prompt_preview(prompt: ChatPromptTemplate, query: str, retriever):
    """
    Show the fully rendered prompt that would be sent to the LLM.
    Useful for debugging — lets you see exactly what context was retrieved
    and how it was inserted into the prompt template.

    Args:
      prompt    : the active ChatPromptTemplate
      query     : the user's question string
      retriever : vector store retriever to fetch context chunks
    """
    docs     = retriever.invoke(query)       # Retrieve relevant chunks
    context  = format_docs(docs)             # Format chunks into a single context string
    rendered = prompt.format(context=context, question=query)  # Fill the template
    print("\n" + "─" * 50)
    print("RENDERED PROMPT PREVIEW:")
    print("─" * 50)
    # Truncate to 1500 chars to avoid flooding the terminal
    print(rendered[:1500] + ("..." if len(rendered) > 1500 else ""))
    print("─" * 50 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE SETUP
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 50)
print("GENERATION PIPELINE")
print("Embeddings : HuggingFace")
print("LLM        : Ollama llama3.2:1b")
print("=" * 50 + "\n")

# ── STEP 1: Load & Split Documents ─────────────────────────────────────────
# Calls load_and_split_docs() from the retrieval module which:
#   - Opens each PDF and extracts text page by page
#   - Cleans and filters pages
#   - Splits pages into 500-char overlapping chunks
print("[1/5] Loading and splitting documents...")
splits, files_to_index = load_and_split_docs()
print(f"  ✔ Loaded {len(files_to_index)} document(s) | {len(splits)} chunks created\n")

# ── STEP 2: Embedding Model ─────────────────────────────────────────────────
# The actual embedding model is initialized inside build_stores().
# This step just confirms the model name that will be used.
print("[2/5] Initializing HuggingFace embedding model...")
print(f"  ✔ Embedding model ready : {HF_MODEL}\n")

# ── STEP 3: Build Vector Stores ─────────────────────────────────────────────
# Calls build_stores() from the retrieval module which:
#   - Initializes HuggingFace embeddings (all-MiniLM-L6-v2, 384-dim)
#   - Indexes all chunks into InMemoryVectorStore with incremental tracking
#   - Attempts to connect to PGVector for persistent storage
print("[3/5] Building vector stores (InMemory + PGVector)...")
embeddings, vectorstore, pg_vectorstore = build_stores(splits, files_to_index)
pg_status = "connected" if pg_vectorstore else "skipped (not available)"
print(f"  ✔ InMemoryVectorStore  : indexed {len(splits)} chunks")
print(f"  ✔ PGVector             : {pg_status}\n")

# ── STEP 4: Build Retriever ─────────────────────────────────────────────────
# Wraps the vector store in a retriever interface.
# Prefers PGVector if available (persistent), falls back to InMemory.
# k=6 means the retriever returns the 6 most relevant chunks per query.
print("[4/5] Building retriever...")
retriever, active_store = build_retriever(vectorstore, pg_vectorstore, k=6)
print(f"  ✔ Active store : {active_store} | top-k : 6\n")

# ── STEP 5: Initialize LLM ──────────────────────────────────────────────────
# ChatOllama connects to the locally running Ollama server.
# llama3.2:1b is used because it fits in ~1.3 GiB RAM (smaller than llama3).
# temperature=0 makes responses deterministic (no randomness).
print("[5/5] Initializing Ollama LLM...")
llm = ChatOllama(model="llama3.2:1b", temperature=0)
print(f"  ✔ LLM ready : llama3.2:1b\n")

# ── Summary ─────────────────────────────────────────────────────────────────
print("=" * 50)
print(f"  Pipeline setup complete!")
print(f"  Documents  : {len(files_to_index)}")
print(f"  Chunks     : {len(splits)}")
print(f"  Embeddings : {HF_MODEL}")
print(f"  Retriever  : {active_store} (k=6)")
print(f"  LLM        : llama3.2:1b")
print("=" * 50 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# INTERACTIVE GENERATION LOOP
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 50)
print("Generation Ready! Commands:")
print("  prompt            — switch prompt type")
print("  preview:<query>   — show rendered prompt before answering")
print("  debug:<query>     — show retrieved chunks without generating")
print("  stream:<query>    — stream answer token by token")
print("  batch:<q1>|<q2>   — batch multiple queries (pipe-separated)")
print("  <query>           — standard generate answer")
print("  exit              — quit")
print("=" * 50 + "\n")

# Start with the default prompt template
current_prompt, current_label = build_prompt("default"), "default"

# Build the initial RAG chain with the default prompt
chain = build_generation_chain(retriever, current_prompt, llm)
print(f"  Active prompt : {current_label}\n")

while True:
    user_input = input("Query: ").strip()

    if user_input.lower() in ("exit", "quit"):
        print("Exiting generation pipeline.")
        break

    if not user_input:
        continue

    # ── Command: switch prompt type ─────────────────────────────────────────
    if user_input.lower() == "prompt":
        # Let user pick a new prompt type from the menu
        current_prompt, current_label = select_prompt()
        # Rebuild the chain with the new prompt
        chain = build_generation_chain(retriever, current_prompt, llm)
        print(f"  Prompt switched to : {current_label}\n")
        continue

    # ── Command: preview rendered prompt ────────────────────────────────────
    if user_input.startswith("preview:"):
        # Show the fully filled prompt without sending it to the LLM
        query = user_input[8:].strip()
        show_prompt_preview(current_prompt, query, retriever)
        continue

    # ── Command: debug retrieved chunks ─────────────────────────────────────
    if user_input.startswith("debug:"):
        # Show which chunks would be retrieved for this query
        query = user_input[6:].strip()
        docs  = retriever.invoke(query)
        print(f"\n  Retrieved {len(docs)} chunks for: '{query}'")
        for i, doc in enumerate(docs):
            print(f"  [{i+1}] {doc.metadata.get('filename','?')} | Page {doc.metadata.get('page','?')}")
            print(f"       {doc.page_content[:200]!r}\n")
        continue

    # ── Command: stream answer token by token ───────────────────────────────
    if user_input.startswith("stream:"):
        # Uses chain.stream() — tokens appear in real time as LLM generates them
        query = user_input[7:].strip()
        print(f"\n  [Prompt: {current_label}] Streaming answer...")
        generate_stream(chain, query)
        print("-" * 50 + "\n")
        continue

    # ── Command: batch multiple queries ─────────────────────────────────────
    if user_input.startswith("batch:"):
        # Uses chain.batch() — all queries processed together in one call
        # Queries are separated by pipe character: batch:q1|q2|q3
        raw     = user_input[6:].strip()
        queries = [q.strip() for q in raw.split("|") if q.strip()]
        if not queries:
            print("  [ERROR] No queries found. Use: batch:query1|query2|query3\n")
            continue
        print(f"\n  [Prompt: {current_label}] Batch generation...")
        generate_batch(chain, queries)
        print("-" * 50 + "\n")
        continue

    # ── Default: standard single query generation ───────────────────────────
    # Uses chain.invoke() — synchronous, waits for full answer before printing
    print(f"\n  [Prompt: {current_label}] Generating answer...")
    answer = chain.invoke(user_input)
    print(f"\nAnswer:\n{answer}\n")
    print("-" * 50 + "\n")
