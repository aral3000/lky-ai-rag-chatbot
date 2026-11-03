"""
LLM Client & Persona Dispatcher for LKY-AI Chatbot.
Supports Local LLM Vault (DPAPI multi-provider auto-failover), direct Gemini/OpenAI (stdlib HTTP),
and Grounded Deterministic Persona Synthesis.
Author: M. Athallah Rafif Aldera, S.Kom.
"""

import os
import sys
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Tuple

# Try attaching local LLM vault from workspace if available
VAULT_DIR = r"c:\new era code\llm-vault"
HAS_LOCAL_VAULT = False
if os.path.isdir(VAULT_DIR):
    if VAULT_DIR not in sys.path:
        sys.path.insert(0, VAULT_DIR)
    try:
        from hub import ask_llm
        HAS_LOCAL_VAULT = True
    except Exception:
        HAS_LOCAL_VAULT = False

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
    provider: str = "Local LLM Vault (Auto-Failover)"
) -> str:
    """
    Generates grounded response using selected provider or fallback persona synthesis.
    All external HTTP calls have strict timeouts (5.0s max) to prevent background hangs.
    """
    prompt = (
        f"Historical Source Context:\n{context_str}\n\n"
        f"User Inquiry: {query}\n\n"
        f"Respond directly in the distinctive voice of Lee Kuan Yew, citing the historical policy precedents when applicable."
    )

    # 1. Local LLM Vault (Primary: Auto-failover across Groq, Cohere, OpenRouter, SambaNova, etc.)
    if ("Vault" in provider or provider == "auto") and HAS_LOCAL_VAULT:
        try:
            result = ask_llm(
                prompt=prompt,
                provider="auto",
                system_prompt=LKY_SYSTEM_PROMPT,
                max_tokens=600,
                temperature=0.3,
                timeout=4.5,
                return_meta=True
            )
            provider_tag = f"*[Synthesized via {result['provider'].upper()} ({result.get('model', '')}) in {result.get('latency', 0):.2f}s]*\n\n"
            return provider_tag + result["text"]
        except Exception as e:
            fallback = _offline_persona_synthesis(query, retrieved_docs)
            return f"> *[Vault Notice: {str(e)[:120]}. Falling back to offline grounded persona engine.]*\n\n{fallback}"

    # 2. Google Gemini Provider via stdlib HTTP (Gemini 1.5 Flash)
    if "Gemini" in provider and api_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key.strip()}"
            payload = {
                "system_instruction": {"parts": [{"text": LKY_SYSTEM_PROMPT}]},
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 600}
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return "*[Synthesized via Google Gemini 1.5 Flash]*\n\n" + text
        except Exception as e:
            fallback = _offline_persona_synthesis(query, retrieved_docs)
            return f"> *[Gemini API Error: {str(e)[:120]}. Falling back to offline grounded persona engine.]*\n\n{fallback}"

    # 3. OpenAI / Compatible Provider via stdlib HTTP
    if "OpenAI" in provider and api_key:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": LKY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 600
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key.strip()}"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data["choices"][0]["message"]["content"]
                return "*[Synthesized via OpenAI GPT-4o-mini]*\n\n" + text
        except Exception as e:
            fallback = _offline_persona_synthesis(query, retrieved_docs)
            return f"> *[OpenAI API Error: {str(e)[:120]}. Falling back to offline grounded persona engine.]*\n\n{fallback}"

    # 4. Deterministic Grounded Persona Engine (Runs 100% offline with zero dependencies)
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
