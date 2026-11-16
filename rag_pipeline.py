import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.messages import HumanMessage, AIMessage

# Configure logging to ignore multiquery internal logs if desired
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Configuration
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:7b"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def process_and_index_pdf(file_path):
    """Loads a PDF, splits it, and indexes it into Chroma."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    return len(chunks)

def query_advanced(question, k=4):
    """Uses MultiQueryRetriever to get context from the vector database."""
    if not os.path.exists(CHROMA_PATH):
        return []
        
    embeddings = get_embeddings()
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # We use a base LLM (with 0 temp) to generate the multi-queries
    llm_for_queries = OllamaLLM(model=LLM_MODEL, temperature=0)
    
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": k}),
        llm=llm_for_queries
    )
    
    unique_docs = retriever_from_llm.invoke(question)
    return unique_docs

def get_chat_history_str(chat_history):
    history_str = ""
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
    return history_str

def query_llm_stream(question, context_docs, chat_history, temperature=0.7):
    """Streams the LLM generated answer based on context and history."""
    llm = OllamaLLM(model=LLM_MODEL, temperature=temperature)
    
    template = """
    You are an advanced AI assistant tasked with answering questions based on the provided context.
    Use the chat history for conversational context if needed.
    If the answer is not in the context, say "I don't know based on the provided documents."
    
    === Chat History ===
    {chat_history}
    
    === Context ===
    {context}
    
    === Current Question ===
    {question}
    
    Answer:
    """
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    
    # Combine context into a single string with page numbers
    context_text = ""
    for doc in context_docs:
        page_num = doc.metadata.get("page", "Unknown")
        context_text += f"[Page {page_num}]\n{doc.page_content}\n\n"
        
    history_text = get_chat_history_str(chat_history[-4:]) # Pass last 4 interactions
    
    # Return a generator for streaming
    return chain.stream({
        "context": context_text, 
        "question": question,
        "chat_history": history_text
    })
