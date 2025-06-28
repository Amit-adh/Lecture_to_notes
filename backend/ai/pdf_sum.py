from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

# Load and split PDF
pdf_path = input("Enter PDF path: ")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
texts = [chunk.page_content for chunk in chunks]

# Use LangChain-compatible embedding model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(texts=texts, embedding=embedder)

# Summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

while True:
    query = input("Your question (or type 'exit'): ")
    if query.lower() == 'exit':
        break

    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    summary = summarizer(context, max_length=200, min_length=30, do_sample=False)[0]['summary_text']

    print("\n--- Summary ---\n")
    print(summary)
    print("\n---------------\n")
