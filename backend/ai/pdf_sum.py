import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def summarize_pdf():
    load_dotenv()

    # === Load PDF ===
    pdf_path = input("📄 Enter path to your PDF: ")
    if not os.path.exists(pdf_path):
        print("❌ File does not exist.")
        return

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages")

    # === Chunk PDF ===
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"🧩 Split into {len(chunks)} text chunks")

    if not chunks:
        print("❌ No text found in PDF. It might be scanned or image-based.")
        return

    # === Create Embeddings (using Ollama locally) ===
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # === Load LLM (Llama3.2 via Ollama) ===
    llm = ChatOllama(model="llama3.2")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # === Prompt Template ===
    prompt_template = """[INST] Use the context to answer the question concisely:
    {context}

    Question: {question} [/INST]"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # === RAG Chain ===
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # === Q&A Loop ===
    print("\n✅ Ready. Ask any question about the PDF (type 'exit' to quit):")
    while True:
        question = input("❓ Question: ")
        if question.lower() == "exit":
            break
        print("🧠 Generating answer...\n")
        print(rag_chain.invoke(question))
        print("-" * 50)

if __name__ == "__main__":
    summarize_pdf()
