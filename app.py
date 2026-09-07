import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configuration
CHROMA_DB_DIR = "./chroma_db"
LLM_MODEL = "qwen2.5-coder:14b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

st.set_page_config(page_title="RFID Coding Assistant", page_icon="📦", layout="wide")

st.title("📦 RFID Coding Assistant")
st.markdown(f"**Powered by {LLM_MODEL}** | *Ask me anything about the RFID Inventory Management Software architecture and code.*")

@st.cache_resource
def load_rag_chain():
    # 1. Load Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # 2. Load Vector Store
    if not os.path.exists(CHROMA_DB_DIR):
        st.error("Chroma DB not found! Please run the ingestion script first.")
        st.stop()
        
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 3. Setup LLM
    llm = OllamaLLM(model=LLM_MODEL)
    
    # 4. Create Prompt Template
    system_prompt = (
        "You are an expert software engineer and assistant for a proprietary RFID inventory management system. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Keep your answers concise and provide code examples where appropriate.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 5. Create Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

try:
    with st.spinner("Loading AI model and knowledge base..."):
        rag_chain = load_rag_chain()
except Exception as e:
    st.error(f"Error loading RAG chain: {e}")
    st.stop()

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input from user
if prompt := st.chat_input("Ask a question about the RFID system..."):
    # Display user msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                sources = response.get("context", [])
                
                st.markdown(answer)
                
                # Show sources if any
                if sources:
                    with st.expander("Sources"):
                        for i, doc in enumerate(sources):
                            source_path = doc.metadata.get('source', 'Unknown source')
                            st.markdown(f"**Source {i+1}:** `{source_path}`")
                            # Optionally show a small snippet
                            # st.markdown(f"```text\n{doc.page_content[:200]}...\n```")
                            
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
                answer = "Sorry, I encountered an error."

    st.session_state.messages.append({"role": "assistant", "content": answer})
