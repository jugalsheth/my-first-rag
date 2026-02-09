"""
Use real LLM (Gemini) for research when GEMINI_API_KEY is set.
Runs RobustRAG error handling with a live Gemini backend and logging.
Without the key, prints instructions and exits (no fake data).
"""

from __future__ import annotations

import os
import sys

# Try to load env so GEMINI_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def make_gemini_query_fn():
    """Build query_fn that calls Gemini (real LLM). Raises on API/rate/timeout errors."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai
    except ImportError:
        print("Install: pip install google-generativeai", file=sys.stderr)
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemma-3-4b-it")  # or gemma-3-12b-it

    def query(question: str) -> str:
        prompt = f"Answer in 1–2 sentences: {question}"
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 150},
        )
        return response.text.strip()

    return query


def main():
    print("Research run with real LLM (Gemini)")
    print("=" * 50)

    query_fn = make_gemini_query_fn()
    if not query_fn:
        print("GEMINI_API_KEY not set. For research using the real LLM:")
        print("  1. Add GEMINI_API_KEY=your_key to .env")
        print("  2. Run: python3 run_with_llm.py")
        print("Tests (test_error_handling.py) use simulated backends so they run without a key.")
        sys.exit(1)

    from robust_rag import RobustRAG

    # Real LLM backend + logging so you see each step
    robust = RobustRAG(query_fn=query_fn, cache={}, log_fn=print)

    questions = [
        "What is RAG in one sentence?",
        "What is retrieval-augmented generation?",
    ]
    for q in questions:
        print(f"\nQuery: {q}")
        r = robust.query(q)
        print(f"Answer: {r.answer[:200]}{'...' if len(r.answer) > 200 else ''}")
        print(f"  from_cache={r.from_cache}, attempts={r.attempts}, circuit_open={r.circuit_open}")

    print(f"\nStats: {robust.stats}")
    print("Done (real Gemini backend).")


if __name__ == "__main__":
    main()
