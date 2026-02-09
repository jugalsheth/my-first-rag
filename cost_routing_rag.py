"""
Day 18/90: Cost routing by query complexity
Routes simple queries → Gemma 4B (cheap), complex → Gemma 12B (smart).
Tracks cost for baseline (all 12B) vs optimized (routed) runs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from query_complexity import classify_query, classify_with_reason, route_model

# Cost per 1K tokens (example: 4B ~1/3 of 12B for input, similar ratio for output)
# Adjust to your provider (e.g. Gemini pricing for gemma-3-4b-it vs gemma-3-12b-it)
DEFAULT_COST_4B = {
    "input_per_1k_tokens": 0.00015,
    "output_per_1k_tokens": 0.0006,
}
DEFAULT_COST_12B = {
    "input_per_1k_tokens": 0.00045,
    "output_per_1k_tokens": 0.0018,
}


@dataclass
class QueryResult:
    """Result of one routed query with cost and model used."""
    query: str
    complexity: str  # "simple" | "complex"
    model_used: str  # "gemma_4b" | "gemma_12b"
    input_tokens: int
    output_tokens: int
    cost: float
    reason: str = ""


class CostRoutingRAG:
    """
    RAG wrapper that routes by query complexity and tracks cost.
    Simple → 4B, Complex → 12B. Supports real Gemini or simulated token counts.
    """

    def __init__(
        self,
        generate_fn: Optional[Callable[[str, str, List[str]], Tuple[str, int, int]]] = None,
        cost_4b: Optional[Dict[str, float]] = None,
        cost_12b: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            generate_fn: (query, model_name, chunks) -> (answer, input_tokens, output_tokens).
                         If None, query() will use simulate_tokens() for testing.
            cost_4b: Per-1k-token costs for 4B. Defaults to DEFAULT_COST_4B.
            cost_12b: Per-1k-token costs for 12B. Defaults to DEFAULT_COST_12B.
        """
        self.generate_fn = generate_fn
        self.cost_4b = {**DEFAULT_COST_4B, **(cost_4b or {})}
        self.cost_12b = {**DEFAULT_COST_12B, **(cost_12b or {})}
        self.results: List[QueryResult] = []
        self.total_cost = 0.0

    def _cost_for_tokens(self, model: str, input_tokens: int, output_tokens: int) -> float:
        costs = self.cost_4b if model == "gemma_4b" else self.cost_12b
        return (input_tokens / 1000.0) * costs["input_per_1k_tokens"] + (
            output_tokens / 1000.0
        ) * costs["output_per_1k_tokens"]

    @staticmethod
    def simulate_tokens(query: str, chunks: List[str], max_context: int = 1200, max_output: int = 200) -> Tuple[int, int]:
        """Simulate input/output token counts for testing (no LLM call). ~4 chars/token."""
        context = "\n\n".join(chunks) if chunks else ""
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        in_tok = min(max(1, len(prompt) // 4), max_context)
        # Simple queries tend shorter answers
        out_tok = max(50, min(max_output, len(query) // 2))
        return in_tok, out_tok

    def query(
        self,
        question: str,
        chunks: Optional[List[str]] = None,
        use_routing: bool = True,
    ) -> Tuple[str, QueryResult]:
        """
        Run one query. If use_routing=True, route by complexity; else use 12B (baseline).
        Returns (answer_text, QueryResult with cost).
        """
        chunks = chunks or []
        complexity, reason = classify_with_reason(question)
        if use_routing:
            model = "gemma_4b" if complexity == "simple" else "gemma_12b"
        else:
            model = "gemma_12b"

        if self.generate_fn:
            answer, in_tok, out_tok = self.generate_fn(question, model, chunks)
        else:
            in_tok, out_tok = self.simulate_tokens(question, chunks)
            answer = f"[simulated answer for {model}]"

        cost = self._cost_for_tokens(model, in_tok, out_tok)
        self.total_cost += cost
        result = QueryResult(
            query=question,
            complexity=complexity,
            model_used=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost=cost,
            reason=reason,
        )
        self.results.append(result)
        return answer, result

    def get_summary(self) -> Dict:
        """Summary: total cost, by model, simple vs complex counts."""
        by_model: Dict[str, Dict] = {"gemma_4b": {"count": 0, "cost": 0.0}, "gemma_12b": {"count": 0, "cost": 0.0}}
        for r in self.results:
            by_model[r.model_used]["count"] += 1
            by_model[r.model_used]["cost"] += r.cost
        return {
            "total_queries": len(self.results),
            "total_cost": round(self.total_cost, 4),
            "by_model": {k: {"count": v["count"], "cost": round(v["cost"], 4)} for k, v in by_model.items()},
        }


def run_baseline_cost(queries: List[str], chunks_per_query: int = 3, ctx_tokens: int = 400) -> float:
    """Simulate baseline: every query uses 12B. Returns total cost."""
    router = CostRoutingRAG()
    for q in queries:
        fake_chunks = ["chunk " * (ctx_tokens // 6)] * chunks_per_query
        router.query(q, chunks=fake_chunks, use_routing=False)
    return round(router.total_cost, 4)


def run_routed_cost(queries: List[str], chunks_per_query: int = 3, ctx_tokens: int = 400) -> Tuple[float, CostRoutingRAG]:
    """Simulate optimized: route by complexity. Returns (total_cost, router)."""
    router = CostRoutingRAG()
    for q in queries:
        fake_chunks = ["chunk " * (ctx_tokens // 6)] * chunks_per_query
        router.query(q, chunks=fake_chunks, use_routing=True)
    return round(router.total_cost, 4), router


def make_gemini_generate_fn(api_key: Optional[str] = None):
    """
    Build generate_fn for CostRoutingRAG that calls Gemini with Gemma 4B or 12B.
    (query, model_name, chunks) -> (answer, input_tokens, output_tokens).
    Requires: pip install google-generativeai, GEMINI_API_KEY.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("google-generativeai required for make_gemini_generate_fn")

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY or api_key required")
    genai.configure(api_key=key)

    model_4b = genai.GenerativeModel("models/gemma-3-4b-it")
    model_12b = genai.GenerativeModel("models/gemma-3-12b-it")

    def generate(question: str, model_name: str, chunks: List[str]) -> Tuple[str, int, int]:
        model = model_4b if model_name == "gemma_4b" else model_12b
        context = "\n\n".join(chunks) if chunks else "No context."
        prompt = f"Based on the context, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        # Approx tokens: ~4 chars/token
        in_tok = max(1, len(prompt) // 4)
        resp = model.generate_content(prompt, generation_config={"temperature": 0.3, "max_output_tokens": 300})
        out_tok = max(1, len(resp.text) // 4)
        return resp.text.strip(), in_tok, out_tok

    return generate
