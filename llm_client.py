"""
LLM Client & Persona Dispatcher for LKY-AI Chatbot.
Supports Google Gemini, OpenAI, Groq, and a Grounded Deterministic Persona Engine.
Author: M. Athallah Rafif Aldera, S.Kom.
"""

import os
from typing import Dict, Any, List, Tuple

LKY_SYSTEM_PROMPT = """You are Lee Kuan Yew, Founding Prime Minister of Singapore (1923 - 2015).
You respond to queries regarding leadership, urban planning, housing, economic development, education, and geopolitics strictly through your pragmatic, analytical, and visionary worldview.

Persona Directives:
1. Tone: Pragmatic, decisive, articulate, realistic, and highly strategic. No sentimental platitudes.
2. Core Ideology: Ruthless meritocracy, rule of law, long-term national viability, racial harmony, public home ownership (HDB/CPF), and zero tolerance for corruption.
3. Grounding: When context sources are provided, reference the historical Singapore policies and experiences directly.
4. Language: Formal, articulate English with crisp argumentation."""


def generate_lky_response(
    query: str,
    context_str: str,
    retrieved_docs: List[Tuple[Dict[str, Any], float]],
    api_key: str = "",
    provider: str = "Demo Mode (Offline)"
) -> str:
    """Generates grounded response using selected provider or fallback persona synthesis."""
    
    # 1. Google Gemini Provider
    if "Gemini" in provider and api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = (
                f"{LKY_SYSTEM_PROMPT}\n\n"
                f"Historical Source Context:\n{context_str}\n\n"
                f"User Inquiry: {query}\n\n"
                f"Respond directly in the distinctive voice of Lee Kuan Yew, citing the historical policy precedents when applicable."
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"[Gemini API Error: {str(e)}]\n\nFalling back to grounded offline persona synthesis below:\n\n" + _offline_persona_synthesis(query, retrieved_docs)

    # 2. OpenAI / Compatible Provider
    if "OpenAI" in provider and api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"{LKY_SYSTEM_PROMPT}\n\nHistorical Context:\n{context_str}"},
                    {"role": "user", "content": query}
                ],
                temperature=0.3
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"[OpenAI API Error: {str(e)}]\n\nFalling back to grounded offline persona synthesis below:\n\n" + _offline_persona_synthesis(query, retrieved_docs)

    # 3. Deterministic Grounded Persona Engine (Runs 100% offline with zero dependencies)
    return _offline_persona_synthesis(query, retrieved_docs)


def _offline_persona_synthesis(query: str, retrieved_docs: List[Tuple[Dict[str, Any], float]]) -> str:
    """
    Synthesizes a pragmatic, authoritative Lee Kuan Yew response grounded directly
    in the retrieved historical speech excerpts.
    """
    if not retrieved_docs:
        return (
            f"Regarding '{query}', you must approach this with unvarnished realism. "
            f"The world does not owe anyone a living. First, analyze the empirical facts, not sentimental wishes. "
            f"Second, build institutions anchored on integrity, competence, and ruthless meritocracy."
        )

    top_doc, score = retrieved_docs[0]
    theme = top_doc.get("theme", "Governance")
    title = top_doc.get("title", "")
    year = top_doc.get("year", "")
    content = top_doc.get("content", "")

    # Tailored contextual openings based on theme
    if "Housing" in theme or "Real Estate" in theme:
        opening = (
            f"On the subject of '{query}', our experience in building modern Singapore provides a definitive answer: "
            f"citizens must have a tangible stake in their nation. When we formulated the '{title}' ({year}), "
            f"we knew that property ownership is the bedrock of social stability."
        )
    elif "Technology" in theme or "Human Capital" in theme:
        opening = (
            f"Technological transition is not a theoretical debate; it is an existential imperative. "
            f"When confronting '{query}', remember that automation and artificial intelligence will not wait for laggards. "
            f"As I argued in {year}, we must relentlessly upgrade human capital."
        )
    elif "Economy" in theme:
        opening = (
            f"In economic statecraft, sentimentality is fatal. Addressing '{query}' requires total credibility and predictability. "
            f"As demonstrated in our policy on '{title}' ({year}), global capital goes only where the rule of law is inviolable and corruption is stamped out."
        )
    else:
        opening = (
            f"To address '{query}', you must strip away dogmatism and look at practical outcomes. "
            f"In {year}, when addressing '{title}', I made it unequivocally clear that survival depends on meritocracy and decisive execution."
        )

    body = (
        f"{opening}\n\n"
        f"Consider our historical precedent:\n"
        f"> *\"{content}\"*\n\n"
        f"The lesson is simple: do not be distracted by theoretical abstractions. "
        f"Put in place the best people, reward performance without compromise, and measure success by real, sustainable results on the ground."
    )

    return body
