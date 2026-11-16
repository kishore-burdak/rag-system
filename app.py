import streamlit as st
import os
import tempfile
import time
import socket
import subprocess
from database_manager import add_stat, get_all_stats
from rag_pipeline import process_and_index_pdf, query_advanced, query_llm_stream

st.set_page_config(page_title="Advanced PDF RAG", layout="wide")
st.title("Advanced Conversational RAG with Qwen 2.5")

@st.cache_resource
def start_ollama():
    """Checks if Ollama is running, and starts it in the background if not."""
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=1):
            return "Ollama is already running."
    except OSError:
        # Start Ollama as a background process
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3) # Wait for server to initialize
        return "Started Ollama server automatically."

# Ensure Ollama is running
start_message = start_ollama()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Upload and Stats
with st.sidebar:
    st.header("LLM Settings")
    temperature = st.slider("Temperature (Creativity)", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    st.divider()
    
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if st.button("Train / Index PDF") and uploaded_file is not None:
        with st.spinner("Processing PDF (Extracting, Splitting, Embedding)..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
                
            file_size_kb = len(uploaded_file.getvalue()) / 1024
            
            try:
                num_chunks = process_and_index_pdf(tmp_path)
                add_stat(uploaded_file.name, round(file_size_kb, 2), num_chunks)
                st.success(f"Successfully processed {uploaded_file.name}!")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
            finally:
                os.remove(tmp_path)
                
    st.divider()
    
    st.header("Memory (Trained PDFs)")
    stats = get_all_stats()
    if not stats:
        st.info("No PDFs have been trained yet.")
    else:
        for stat in stats:
            st.markdown(f"**{stat['filename']}**")
            st.caption(f"Size: {stat['size_kb']} KB | Chunks: {stat['num_chunks']}")
            st.progress(1.0)
            
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Multi-Query Retrieval in progress..."):
            context_docs = query_advanced(prompt)
            
        if not context_docs:
            st.warning("No context found. Please ensure a PDF is trained.")
            st.session_state.messages.append({"role": "assistant", "content": "No context found. Please ensure a PDF is trained."})
        else:
            # Show citations expander
            with st.expander("View Retrieved Context & Citations"):
                for i, doc in enumerate(context_docs):
                    page = doc.metadata.get('page', 'N/A')
                    st.markdown(f"**Source {i+1} (Page {page})**:")
                    st.caption(doc.page_content)
                    
            # Stream response
            try:
                stream_generator = query_llm_stream(
                    prompt, 
                    context_docs, 
                    st.session_state.messages, 
                    temperature=temperature
                )
                response = st.write_stream(stream_generator)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error communicating with local LLM (ensure ollama is running): {e}")
