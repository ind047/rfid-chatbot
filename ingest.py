import os
import subprocess
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

REPO_URL = "https://github.com/zetavg/Inventory.git"
REPO_DIR = "rfid_repo"
CHROMA_DB_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def clone_repo():
    if not os.path.exists(REPO_DIR):
        print(f"Cloning repository {REPO_URL}...")
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    else:
        print("Repository already cloned.")

def load_documents():
    print("Loading documents from repository...")
    documents = []
    
    # Load Java files
    java_loader = DirectoryLoader(REPO_DIR, glob="**/*.java", loader_cls=TextLoader)
    documents.extend(java_loader.load())
    
    # Load TS/TSX files
    ts_loader = DirectoryLoader(REPO_DIR, glob="**/*.ts", loader_cls=TextLoader, use_multithreading=True, silent_errors=True)
    documents.extend(ts_loader.load())
    tsx_loader = DirectoryLoader(REPO_DIR, glob="**/*.tsx", loader_cls=TextLoader, use_multithreading=True, silent_errors=True)
    documents.extend(tsx_loader.load())

    # Load JS/JSX files
    js_loader = DirectoryLoader(REPO_DIR, glob="**/*.js", loader_cls=TextLoader, use_multithreading=True, silent_errors=True)
    documents.extend(js_loader.load())
    jsx_loader = DirectoryLoader(REPO_DIR, glob="**/*.jsx", loader_cls=TextLoader, use_multithreading=True, silent_errors=True)
    documents.extend(jsx_loader.load())

    # READMEs and Markdown
    md_loader = DirectoryLoader(REPO_DIR, glob="**/*.md", loader_cls=TextLoader, use_multithreading=True, silent_errors=True)
    documents.extend(md_loader.load())
    
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents):
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def build_vector_store(chunks):
    print("Building vector store (this may take a minute depending on model)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    print(f"Vector store persisted to {CHROMA_DB_DIR}")

def main():
    clone_repo()
    docs = load_documents()
    if not docs:
        print("No documents found to ingest!")
        return
    chunks = split_documents(docs)
    build_vector_store(chunks)
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
