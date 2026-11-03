"""
What Would Lee Kuan Yew Do? (LKY-AI)
Interactive RAG Conversational AI Persona Chatbot
Built for 99 Group Graduate AI Assessment
Author: M. Athallah Rafif Aldera, S.Kom.
"""

import os
import streamlit as st
from rag_engine import LKYVectorStore
from llm_client import generate_lky_response, LKY_SYSTEM_PROMPT

# Streamlit Page Config
st.set_page_config(
    page_title="What Would Lee Kuan Yew Do? (LKY-AI)",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Singapore Crimson & Modern Dark Aesthetic)
st.markdown("""
<style>
    .main-header {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #d32f2f;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #888888;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }
    .citation-box {
        background-color: rgba(211, 47, 47, 0.08);
        border-left: 3px solid #d32f2f;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.82rem;
        margin-top: 8px;
    }
    .badge-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        background: #1e293b;
        color: #f1f5f9;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize RAG Engine
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "lky_corpus.json")

@st.cache_resource
def get_vector_store():
    return LKYVectorStore(CORPUS_PATH)

store = get_vector_store()

# Sidebar Setup
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    model_provider = st.selectbox(
        "Model Provider",
        ["Demo Mode (Offline - Grounded Synthesis)", "Google Gemini (gemini-1.5-flash)", "OpenAI (gpt-4o-mini)"],
        index=0
    )
    api_key = st.text_input(
        "API Key (Optional for Demo Mode)",
        type="password",
        help="Leave blank to use the deterministic offline persona synthesis engine."
    )
    
    st.markdown("---")
    st.markdown("### 🔍 RAG Retrieval Configuration")
    top_k = st.slider("Top-k Documents Retrieved", min_value=1, max_value=4, value=2)
    min_score = st.slider("Relevance Score Threshold", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
    
    st.markdown("---")
    st.markdown("### 📚 Knowledge Corpus Stats")
    st.markdown(f"- **Indexed Documents**: {len(store.documents)}")
    st.markdown(f"- **Unique Vocabulary Terms**: {len(store.vocabulary)}")
    st.markdown(f"- **Primary Focus**: Singapore Housing (HDB), Economy & AI")
    
    st.markdown("---")
    st.caption("Developed by **M. Athallah Rafif Aldera, S.Kom.** for 99 Group Assessment.")

# Main Interface Header
st.markdown('<h1 class="main-header">🦁 What Would Lee Kuan Yew Do?</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Grounded RAG Conversational AI Persona trained on Lee Kuan Yew\'s Speeches, HDB Housing Policies & Memoirs</p>', unsafe_allow_html=True)

# Preset Quick-Prompt Buttons
st.markdown("**Quick Prompts (Click to explore):**")
c1, c2, c3 = st.columns(3)
preset_query = None
with c1:
    if st.button("🏢 HDB & Public Home Ownership", use_container_width=True):
        preset_query = "Why did you prioritize public home ownership through HDB and CPF over public rental housing?"
with c2:
    if st.button("🤖 AI & Technological Disruption", use_container_width=True):
        preset_query = "How should leaders and nations adapt to artificial intelligence and automation without causing massive unemployment?"
with c3:
    if st.button("⚖️ Meritocracy & Anti-Corruption", use_container_width=True):
        preset_query = "What is your stance on eliminating corruption and ensuring meritocratic leadership in governance?"

# Chat History State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "I am Lee Kuan Yew. In statecraft and nation-building, sentimentality is a luxury no viable system can afford. You may ask me about urban planning, public housing, economic transformation, technological adaptation, or governance strategy.",
            "citations": []
        }
    ]

# Render Message History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📚 Retrieved Historical Sources ({len(msg['citations'])} excerpts)"):
                for doc, score in msg["citations"]:
                    st.markdown(f"**{doc.get('title')} ({doc.get('year')})** — `Relevance: {score:.2f}`")
                    st.markdown(f"*Source: {doc.get('source')}*")
                    st.markdown(f"> \"{doc.get('content')}\"")
                    st.divider()

# Input Dispatcher
active_query = preset_query or st.chat_input("Enter your question for Lee Kuan Yew...")

if active_query:
    # Append User Message
    st.session_state.chat_history.append({"role": "user", "content": active_query, "citations": []})
    with st.chat_message("user"):
        st.markdown(active_query)

    # Execute Vector Search & Generation
    with st.chat_message("assistant"):
        with st.spinner("Retrieving historical speeches & formulating strategic response..."):
            retrieved = store.search(active_query, top_k=top_k, min_score=min_score)
            context_str = store.format_retrieved_context(retrieved)
            
            response = generate_lky_response(
                query=active_query,
                context_str=context_str,
                retrieved_docs=retrieved,
                api_key=api_key,
                provider=model_provider
            )
            
            st.markdown(response)
            
            # Display Citations in UI
            if retrieved:
                with st.expander(f"📚 Retrieved Historical Sources ({len(retrieved)} excerpts)"):
                    for doc, score in retrieved:
                        st.markdown(f"**{doc.get('title')} ({doc.get('year')})** — `Relevance: {score:.2f}`")
                        st.markdown(f"*Source: {doc.get('source')}*")
                        st.markdown(f"> \"{doc.get('content')}\"")
                        st.divider()

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "citations": retrieved
        })
