"""
Day 18/90: Query Complexity Classifier
Classifies queries as Simple (factoid, single-hop) vs Complex (multi-hop, reasoning)
for cost-aware model routing: Simple → Gemma 4B (cheap), Complex → Gemma 12B (smart).
"""

from __future__ import annotations

from typing import Literal, Tuple

QueryComplexity = Literal["simple", "complex"]

# Factoid triggers: short questions or what/who/when/where/which → simple
SIMPLE_QUESTION_STARTS = ("what", "who", "when", "where", "which", "is", "are", "did", "does", "can")
# Reasoning / multi-hop triggers → complex
COMPLEX_TRIGGERS = ("compare", "comparison", "analyze", "analysis", "explain", "explanation", "why", "how")
# "why" and "how" at start of sentence → complex (reasoning)
COMPLEX_STARTS = ("why", "how")


def word_count(text: str) -> int:
    """Count words (split on whitespace)."""
    return len(text.strip().split()) if text.strip() else 0


def normalize_for_classification(query: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    return " ".join(query.lower().strip().split())


def classify_query(query: str) -> QueryComplexity:
    """
    Classify query as 'simple' or 'complex' using rule-based heuristics.

    Rules (in order):
    1. Length < 10 words → simple
    2. Contains compare/analyze/explain (anywhere) → complex
    3. Starts with why/how → complex (reasoning)
    4. Starts with what/who/when/where/which (factoid) → simple
    5. Default → complex (unsure cases use smarter model)
    """
    if not query or not query.strip():
        return "simple"

    q = normalize_for_classification(query)
    words = q.split()
    n_words = len(words)

    # Rule 1: Short queries → simple
    if n_words < 10:
        # But still check for complex triggers in short queries (e.g. "Compare X and Y")
        for trigger in COMPLEX_TRIGGERS:
            if trigger in q:
                return "complex"
        if words and words[0] in COMPLEX_STARTS:
            return "complex"
        return "simple"

    # Rule 2: Compare / analyze / explain → complex
    for trigger in COMPLEX_TRIGGERS:
        if trigger in q:
            return "complex"

    # Rule 3: Starts with why/how → complex
    if words and words[0] in COMPLEX_STARTS:
        return "complex"

    # Rule 4: Factoid starters → simple
    if words and words[0] in SIMPLE_QUESTION_STARTS:
        return "simple"

    # Rule 5: Default → complex (prioritize quality for ambiguous)
    return "complex"


def classify_with_reason(query: str) -> Tuple[QueryComplexity, str]:
    """
    Classify and return a short reason string for debugging/audit.
    """
    if not query or not query.strip():
        return "simple", "empty"

    q = normalize_for_classification(query)
    words = q.split()
    n_words = len(words)

    if n_words < 10:
        for trigger in COMPLEX_TRIGGERS:
            if trigger in q:
                return "complex", f"short but contains '{trigger}'"
        if words and words[0] in COMPLEX_STARTS:
            return "complex", f"starts with '{words[0]}' (reasoning)"
        return "simple", f"length < 10 words ({n_words})"

    for trigger in COMPLEX_TRIGGERS:
        if trigger in q:
            return "complex", f"contains '{trigger}'"

    if words and words[0] in COMPLEX_STARTS:
        return "complex", f"starts with '{words[0]}'"

    if words and words[0] in SIMPLE_QUESTION_STARTS:
        return "simple", f"factoid start '{words[0]}'"

    return "complex", "default (ambiguous)"


def route_model(query: str) -> Literal["gemma_4b", "gemma_12b"]:
    """
    Route to model name: simple → gemma_4b (cheap), complex → gemma_12b (smart).
    """
    return "gemma_4b" if classify_query(query) == "simple" else "gemma_12b"
