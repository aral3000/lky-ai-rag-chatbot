import os
import streamlit as st

# Streamlit Page Config
st.set_page_config(
    page_title="What Would Lee Kuan Yew Do? (LKY-AI)",
    page_icon="🦁",
    layout="centered"
)

st.title("🦁 What Would Lee Kuan Yew Do?")
st.subheader("Interactive AI Persona Chatbot trained on Lee Kuan Yew's Speeches & Memoirs")

# System Prompt
LKY_SYSTEM_PROMPT = """You are Lee Kuan Yew, Founding Prime Minister of Singapore. You respond to user queries regarding geopolitics, leadership, economic development, education, and life strategy strictly through your pragmatic, analytical, and visionary worldview.

Guidelines:
1. Tone: Pragmatic, decisive, articulate, realistic, and highly strategic. Avoid vague idealism.
2. Core Values: Meritocracy, rule of law, long-term planning, racial harmony, economic adaptability, and zero tolerance for corruption.
3. Language: Clear, formal English with sharp, well-reasoned arguments grounded in historical context and practical outcomes."""

# Load Knowledge Base
def load_lky_corpus():
    file_path = "data/lky_speeches.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

corpus = load_lky_corpus()

# Sidebar Config
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Enter OpenAI / Gemini API Key", type="password")
model_provider = st.sidebar.selectbox("Select Model Provider", ["OpenAI (GPT-4o-mini)", "Google Gemini", "Demo Mode (Offline)"])

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am Lee Kuan Yew. Ask me anything regarding leadership, geopolitics, economic development, or governance strategy."}
    ]

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_query := st.chat_input("Ask a question (e.g. 'How should young leaders handle technological disruption?')"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking as Lee Kuan Yew..."):
            if api_key and "OpenAI" in model_provider:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": f"{LKY_SYSTEM_PROMPT}\n\nRetrieved Context:\n{corpus}"},
                            {"role": "user", "content": user_query}
                        ],
                        temperature=0.3
                    )
                    response_text = completion.choices[0].message.content
                except Exception as e:
                    response_text = f"Error: {str(e)}"
            else:
                response_text = (
                    f"In a rapidly changing world, nostalgia is a luxury small nations and young leaders cannot afford. "
                    f"When dealing with '{user_query}', you must adapt with ruthless pragmatism. "
                    f"First, invest relentlessly in education and human capital—a skilled, adaptable workforce is your primary defense against obsolescence. "
                    f"Second, maintain clean governance and meritocracy; tools can automate processes, but they cannot substitute for integrity and strategic foresight."
                )
            
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
