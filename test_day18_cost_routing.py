"""
Day 18/90: Test cost optimization via query complexity routing
20 queries: baseline (all 12B) vs routed (simple→4B, complex→12B). Measure cost savings.
"""

from __future__ import annotations

import json
from pathlib import Path

from query_complexity import classify_query, classify_with_reason, route_model
from cost_routing_rag import CostRoutingRAG, run_baseline_cost, run_routed_cost

# 20 test queries: mix of simple (factoid, short) and complex (compare, analyze, explain, why, how)
TEST_QUERIES = [
    # Simple: short or what/who/when
    "What is RAG?",
    "Who wrote the CRAG paper?",
    "When was BERT released?",
    "What is chunk size?",
    "Who invented the transformer?",
    "When to use dense retrieval?",
    "What is embedding?",
    "Which model is best for RAG?",
    "What is top-k?",
    "Who proposed HyDE?",
    # Complex: compare, analyze, explain, why, how
    "Compare dense and sparse retrieval for RAG systems.",
    "Analyze the trade-offs between chunk size and retrieval accuracy.",
    "Explain how self-RAG improves answer quality through iterative refinement.",
    "Why does multi-query retrieval improve coverage?",
    "How does re-ranking improve precision in two-stage retrieval?",
    "Compare CRAG and Self-RAG routing strategies.",
    "Explain the role of uncertainty estimation in corrective RAG.",
    "Why is chunk overlap important in document splitting?",
    "How do hypothetical documents bridge the semantic gap in HyDE?",
    "Analyze when to use local vs web fallback in CRAG.",
]


def main():
    print("Day 18/90: Cost optimization – query complexity routing")
    print("=" * 60)
    print("Classifier rules: length < 10 words = simple; what/who/when = simple; compare/analyze/explain = complex")
    print()

    # Classifier sanity check
    print("Query classification (first 8 + 4 complex):")
    for q in TEST_QUERIES[:8] + TEST_QUERIES[10:14]:
        complexity, reason = classify_with_reason(q)
        model = route_model(q)
        short = (q[:50] + "...") if len(q) > 50 else q
        print(f"  [{complexity:6}] → {model:10}  ({reason})  \"{short}\"")
    print()

    # Baseline: all 12B
    baseline_cost = run_baseline_cost(TEST_QUERIES)
    routed_cost, router = run_routed_cost(TEST_QUERIES)
    savings_dollars = baseline_cost - routed_cost
    savings_pct = (savings_dollars / baseline_cost * 100) if baseline_cost > 0 else 0

    print("Cost comparison (20 queries, simulated tokens):")
    print(f"  Baseline (all Gemma 12B):  ${baseline_cost:.4f}")
    print(f"  Routed (4B/12B by complexity): ${routed_cost:.4f}")
    print(f"  Savings: ${savings_dollars:.4f} ({savings_pct:.1f}%)")
    print()

    summary = router.get_summary()
    print("Routed breakdown:")
    print(f"  Queries to 4B:  {summary['by_model']['gemma_4b']['count']}  (${summary['by_model']['gemma_4b']['cost']:.4f})")
    print(f"  Queries to 12B: {summary['by_model']['gemma_12b']['count']}  (${summary['by_model']['gemma_12b']['cost']:.4f})")
    print()

    # Per-query table (compact)
    print("Per-query routing:")
    for r in router.results:
        q_short = (r.query[:49] + "...") if len(r.query) > 52 else r.query
        print(f"  {r.model_used:10}  ${r.cost:.4f}  {r.complexity:6}  {q_short}")
    print()

    # Save report
    report = {
        "day": 18,
        "description": "Cost optimization via query complexity routing (simple→4B, complex→12B)",
        "num_queries": len(TEST_QUERIES),
        "baseline_cost_all_12b": baseline_cost,
        "routed_cost": routed_cost,
        "savings_dollars": round(savings_dollars, 4),
        "savings_percent": round(savings_pct, 1),
        "routed_breakdown": summary,
        "per_query": [
            {
                "query": r.query,
                "complexity": r.complexity,
                "model_used": r.model_used,
                "cost": round(r.cost, 4),
                "reason": r.reason,
            }
            for r in router.results
        ],
    }
    out_path = Path("cost_routing_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {out_path}")

    return report


if __name__ == "__main__":
    main()
