"""
AI-Powered Industrial Chatbot - Streamlit Web Interface
Integrates with the existing IndustrialChatbot RAG engine (SentenceTransformers + FAISS + Groq/Local LLM).
"""

import sys
import os
from pathlib import Path

import streamlit as st

# Ensure project root is in python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from chatbot import IndustrialChatbot, EMBEDDING_MODEL, INDEX_PATH, METADATA_PATH, RELEVANCE_THRESHOLD


# ============================================================
# PAGE CONFIGURATION & INDUSTRIAL THEME
# ============================================================

st.set_page_config(
    page_title="AI-Powered Industrial Chatbot",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom industrial styling
st.markdown("""
<style>
    /* Main container styling */
    .main-title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    /* Source badges */
    .source-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        font-size: 0.88rem;
    }
    .source-badge {
        display: inline-block;
        background-color: #E2E8F0;
        color: #0F172A;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
    }
    .metric-badge {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# ENGINE CACHING & INITIALIZATION
# ============================================================

@st.cache_resource(show_spinner=False)
def load_chatbot_engine(backend_mode: str) -> IndustrialChatbot:
    """
    Cache the IndustrialChatbot instance to prevent reloading
    SentenceTransformers and FAISS index on every Streamlit rerun.
    """
    return IndustrialChatbot(backend=backend_mode)


# ============================================================
# SIDEBAR CONFIGURATION & CONTROLS
# ============================================================

st.sidebar.markdown("### ⚙️ System Configuration")

# Inference backend selection
backend_option = st.sidebar.radio(
    "Inference Engine",
    options=["Remote Inference (Groq)", "Local Fallback (Hugging Face)"],
    index=0,
    help="Select remote Groq API for rapid answers and low RAM usage, or local CPU Transformers for offline demonstration."
)

if backend_option.startswith("Remote"):
    active_backend = "remote"
    st.sidebar.success("🟢 **Remote Mode Active**: High-speed inference via Groq API (<200 MB RAM).")
else:
    active_backend = "local"
    st.sidebar.warning(
        "⚠️ **High Memory Notice**\n\n"
        "Local CPU mode allocates ~5 GB of RAM for `Qwen2.5-1.5B-Instruct` "
        "and runs significantly slower on CPU."
    )
    confirm_local = st.sidebar.checkbox(
        "Confirm: Load 1.5B model into local RAM",
        value=False,
        help="Check this box to confirm loading the heavy local model."
    )
    if not confirm_local:
        st.sidebar.info("Falling back to Remote mode until local execution is confirmed.")
        active_backend = "remote"

# Initialize the selected engine safely
try:
    with st.spinner(f"Initializing chatbot ({active_backend.upper()} mode)..."):
        bot = load_chatbot_engine(active_backend)
        init_error = None
except Exception as exc:
    bot = None
    init_error = f"Engine initialization failed: {exc}"

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Knowledge Base Info")

if bot and hasattr(bot, "index"):
    st.sidebar.markdown(f"• **Indexed Chunks**: `{bot.index.ntotal}`")
    st.sidebar.markdown(f"• **Relevance Floor**: `{RELEVANCE_THRESHOLD}`")
    if active_backend == "remote":
        model_label = f"{getattr(bot, 'model_id', 'qwen/qwen3.8-27b')} (Remote)"
    else:
        model_label = "Qwen2.5-1.5B-Instruct (Local CPU)"
    st.sidebar.markdown(f"• **Active LLM**: `{model_label}`")

st.sidebar.markdown("• **Technical Documents**:")
documents_list = [
    "Embedded System Design for Heavy-lift UAV",
    "Products and Solutions for Drones",
    "Robotics and Automation - R22",
    "SECA4003 - Embedded Drone Systems",
    "Lecture Slides (Sensors & Avionics)",
    "Module 5 (Actuators & Control)"
]
for doc in documents_list:
    st.sidebar.markdown(f"  - <span style='font-size:0.82rem;'>{doc}</span>", unsafe_allow_html=True)

st.sidebar.markdown("---")

def _get_initial_greeting():
    return [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your **AI Industrial Chatbot**.\n\n"
                "I can help you analyze technical documents about "
                "**drones, robotics, avionics, embedded controllers, and industrial automation**.\n\n"
                "Ask a technical question to explore the knowledge base."
            ),
            "sources": []
        }
    ]

# Clear chat control
if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
    st.session_state.messages = _get_initial_greeting()
    st.rerun()


# ============================================================
# MAIN APPLICATION INTERFACE
# ============================================================

st.markdown('<div class="main-title">AI-Powered Industrial Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Evidence-grounded Question Answering over industrial technical documentation '
    'using Retrieval-Augmented Generation (FAISS + Sentence Transformers).'
    '</div>',
    unsafe_allow_html=True
)

# Display initialization error if occurred
if init_error:
    st.error(f"⚠️ {init_error}")
    st.stop()

# Initialize session state for conversation history
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = _get_initial_greeting()

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display supporting document sources if available
        if msg.get("sources"):
            with st.expander(f"📑 Supporting Document Sources ({len(msg['sources'])} retrieved chunks)", expanded=False):
                for idx, src in enumerate(msg["sources"], 1):
                    sim_val = src.get("similarity")
                    sim_text = f"{sim_val:.4f}" if isinstance(sim_val, (int, float)) else "N/A"
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <span class="source-badge">Source {idx}</span>
                            <strong>Document:</strong> <code>{src.get('source', 'Unknown')}</code><br>
                            <strong>Location:</strong> Page {src.get('page', 'N/A')} &nbsp;|&nbsp;
                            <span class="metric-badge">Cosine Similarity: {sim_text}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ============================================================
# USER INPUT & QUERY PROCESSING
# ============================================================

user_query = st.chat_input("Ask a technical question about UAVs, robotics, or embedded systems...")

if user_query:
    query_text = user_query.strip()
    
    if query_text:
        # Append and display user message
        st.session_state.messages.append({
            "role": "user",
            "content": query_text,
            "sources": []
        })
        
        with st.chat_message("user"):
            st.markdown(query_text)
            
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching technical documents and synthesizing answer..."):
                try:
                    response = bot.answer(query_text)
                    answer_text = response.get("answer", "No response generated.")
                    sources = response.get("sources", [])
                except Exception:
                    answer_text = (
                        "An unexpected error occurred while processing your question. "
                        "Please verify system connectivity and try again."
                    )
                    sources = []
                    
            st.markdown(answer_text)
            
            if sources:
                with st.expander(f"📑 Supporting Document Sources ({len(sources)} retrieved chunks)", expanded=True):
                    for idx, src in enumerate(sources, 1):
                        sim_val = src.get("similarity")
                        sim_text = f"{sim_val:.4f}" if isinstance(sim_val, (int, float)) else "N/A"
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <span class="source-badge">Source {idx}</span>
                                <strong>Document:</strong> <code>{src.get('source', 'Unknown')}</code><br>
                                <strong>Location:</strong> Page {src.get('page', 'N/A')} &nbsp;|&nbsp;
                                <span class="metric-badge">Cosine Similarity: {sim_text}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
            # Save assistant message to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources
            })
