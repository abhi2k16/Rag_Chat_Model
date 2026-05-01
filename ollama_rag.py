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

#PDF_PATH = "LangChain Chat with Your Data/MachineLearning-Lecture01.pdf"
PDF_PATH = "Attention Is All You Need.pdf"
# 1. Load PDF with pdfplumber
print(f"Loading: {PDF_PATH}")
docs = []
with pdfplumber.open(PDF_PATH) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text and text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": PDF_PATH, "page": i + 1}
            ))
print(f"Loaded {len(docs)} pages")

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
