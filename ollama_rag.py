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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

PROJECT_FOLDER = Path(__file__).resolve().parent
PDF_FOLDER = PROJECT_FOLDER / "rag_docs"
PDF_FILES = [str(path) for path in sorted(PDF_FOLDER.glob("*.pdf"))]

# 1. Load PDFs with pdfplumber
if not PDF_FILES:
    raise FileNotFoundError(f"No PDF files found in {PDF_FOLDER}")

print(f"Loading {len(PDF_FILES)} PDF(s) from: {PDF_FOLDER}")
docs = []
for pdf_path in PDF_FILES:
    filename = Path(pdf_path).name
    page_count = 0
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": pdf_path, "filename": filename, "page": i + 1}
                ))
                page_count += 1
    print(f"Loaded {page_count} pages from {filename}")

if not docs:
    raise ValueError(f"No extractable text found in PDFs under {PDF_FOLDER}")

print(f"Loaded {len(docs)} total pages")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)
print(f"Split into {len(splits)} chunks")

# 3. Embed and index into InMemoryVectorStore
print("Indexing chunks (this may take a moment)...")
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("Indexing complete!")

# 4. Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the context below.

Context:
{context}

Question: {question}
""")

# 5. LLM
llm = ChatOllama(model="llama3", temperature=0.3)

# 6. RAG chain
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

# 7. Interactive Q&A loop
print("\nRAG pipeline ready! Type 'exit' to quit.\n")
while True:
    question = input("Question: ").strip()
    if question.lower() in ("exit", "quit"):
        break
    if not question:
        continue
    answer = rag_chain.invoke(question)
    print(f"\nAnswer: {answer}\n")
