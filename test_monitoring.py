"""
Day 17: Test suite for RAG monitoring & observability.
Runs 50 simulated queries (mix cached/uncached), some failures, exports dashboard JSON.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from monitoring import RAGMonitor


def simulate_query(
    monitor: RAGMonitor,
    query: str,
    cached: bool,
    cache_tier: str | None,
    fail: bool = False,
    quality_score: float | None = None,
) -> None:
    """Simulate one RAG query: latency, cost, optional error/quality."""
    if fail:
        latency_ms = random.uniform(500, 3000)  # slow before failure
        cost_usd = 0.0
        monitor.log_query(
            query=query,
            latency_ms=latency_ms,
            cached=False,
            cost_usd=cost_usd,
            quality_score=None,
            error=random.choice(["rate_limit", "timeout", "parse_error"]),
            cache_tier=None,
        )
        return

    if cached:
        latency_ms = random.uniform(0.5, 5.0)  # cache fast
        cost_usd = 0.0
    else:
        latency_ms = random.uniform(200, 1200)  # full RAG
        cost_usd = round(random.uniform(0.0001, 0.001), 6)

    if quality_score is None and not fail:
        quality_score = round(random.uniform(0.6, 0.95), 4)

    monitor.log_query(
        query=query,
        latency_ms=latency_ms,
        cached=cached,
        cost_usd=cost_usd,
        quality_score=quality_score,
        error=None,
        cache_tier=cache_tier,
    )


def main() -> None:
    monitor = RAGMonitor(max_query_history=500)

    # Example queries (repeated to get cache hits)
    queries = [
        "What is RAG?",
        "How does RAG work?",
        "Explain retrieval augmented generation",
        "What are the 3 types of RAG?",
        "Compare RAG to fine-tuning",
        "How to improve RAG retrieval?",
        "What is HyDE?",
        "What is CRAG?",
        "Explain chunking strategies",
        "What is embedding?",
    ]

    # 50 queries: ~34 cached (68%), ~13 uncached, ~3 failures
    random.seed(42)
    for i in range(50):
        q = random.choice(queries)
        r = random.random()
        if r < 0.06:
            simulate_query(monitor, q, cached=False, cache_tier=None, fail=True)
        elif r < 0.68:
            tier = "exact" if random.random() < 0.6 else "semantic"
            simulate_query(monitor, q, cached=True, cache_tier=tier)
        else:
            simulate_query(monitor, q, cached=False, cache_tier=None)

    summary = monitor.get_summary()
    out_dir = Path(__file__).resolve().parent
    dashboard_path = out_dir / "metrics_dashboard.json"
    payload = monitor.export_dashboard(filepath=str(dashboard_path))

    # Console output
    print("Day 17: Monitoring test run")
    print("=" * 50)
    print(f"Total queries:    {summary['total_queries']}")
    print(f"Cache hits:       {summary['cache_hits']}")
    print(f"Cache misses:     {summary['cache_misses']}")
    print(f"Cache hit rate:   {summary['cache_hit_rate']:.2%}")
    print(f"Avg latency:      {summary['avg_latency_ms']:.0f} ms")
    print(f"p50 latency:      {summary['p50_latency_ms']:.0f} ms")
    print(f"p95 latency:      {summary['p95_latency_ms']:.0f} ms")
    print(f"p99 latency:      {summary['p99_latency_ms']:.0f} ms")
    print(f"Total cost:       ${summary['total_cost_usd']:.4f}")
    print(f"Errors:           {summary['error_count']}")
    if summary.get("avg_quality_score") is not None:
        print(f"Avg quality:       {summary['avg_quality_score']:.2f}")
    print("=" * 50)
    print(f"Dashboard exported: {dashboard_path}")
    print("Done.")

    # Assertions for CI / sanity
    assert summary["total_queries"] == 50
    assert 0 <= summary["cache_hit_rate"] <= 1
    assert summary["p95_latency_ms"] >= summary["p50_latency_ms"]
    assert dashboard_path.exists()


if __name__ == "__main__":
    main()
