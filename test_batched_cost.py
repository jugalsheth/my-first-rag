"""
Test batched cost optimization: 2s queue, batch embed, per-query tracking.
20 queries (optionally with repeats for cache). Baseline vs optimized. Generate cost_optimization_results.md.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from batched_cost_pipeline import (
    BatchedCostPipeline,
    PerQueryRecord,
    run_baseline_no_optimization,
    BATCH_WINDOW_SECONDS,
)
from test_day18_cost_routing import TEST_QUERIES


def make_embed_fn(dim: int = 384):
    def _embed(queries):
        rng = np.random.default_rng(42)
        return [rng.random(dim).astype(np.float32).tolist() for _ in queries]
    return _embed


def generate_report_md(
    baseline_cost: float,
    optimized_cost: float,
    baseline_records: list,
    optimized_records: list,
    num_embed_calls_baseline: int,
    num_embed_calls_optimized: int,
    out_path: Path,
) -> None:
    savings_dollars = baseline_cost - optimized_cost
    savings_pct = (savings_dollars / baseline_cost * 100) if baseline_cost > 0 else 0

    by_type = {}
    for r in optimized_records:
        key = f"{r.complexity}_{'cache_hit' if r.cache_hit else 'miss'}"
        if key not in by_type:
            by_type[key] = {"count": 0, "cost": 0.0}
        by_type[key]["count"] += 1
        by_type[key]["cost"] += r.total_cost

    lines = [
        "# Cost Optimization Results",
        "",
        "## Summary",
        "",
        "| Metric | Before (baseline) | After (optimized) |",
        "|--------|--------------------|--------------------|",
        f"| Total cost | ${baseline_cost:.4f} | ${optimized_cost:.4f} |",
        f"| Embedding API calls | {num_embed_calls_baseline} | {num_embed_calls_optimized} |",
        f"| **Savings** | **${savings_dollars:.4f}** | **{savings_pct:.1f}%** |",
        "",
        "**Optimizations applied:**",
        "- **Batching:** Queries queued for 2 seconds, then batch embedded (single API call per window).",
        f"- **Embedding call reduction:** {num_embed_calls_baseline} → {num_embed_calls_optimized} calls ({100 * (1 - num_embed_calls_optimized / max(1, num_embed_calls_baseline)):.0f}% reduction).",
        "- **Model routing:** Simple queries → Gemma 4B (cheap), complex → Gemma 12B (smart).",
        "- **Cache:** Repeated queries served from cache (0 embed + 0 LLM cost).",
        "",
        "---",
        "",
        "## Cost by query type (optimized)",
        "",
        "| Type | Count | Total cost ($) | Avg cost/query ($) |",
        "|------|-------|----------------|---------------------|",
    ]
    for key in sorted(by_type.keys()):
        v = by_type[key]
        avg = v["cost"] / v["count"] if v["count"] else 0
        lines.append(f"| {key} | {v['count']} | {v['cost']:.4f} | {avg:.6f} |")
    lines.extend([
        "",
        "---",
        "",
        "## Per-query detail (optimized)",
        "",
        "| # | Query | Model | Cache hit | Batch size | Total cost ($) |",
        "|---|-------|-------|-----------|------------|----------------|",
    ])
    for i, r in enumerate(optimized_records, 1):
        q_short = (r.query[:48] + "…") if len(r.query) > 50 else r.query
        lines.append(f"| {i} | {q_short} | {r.model_used} | {'Yes' if r.cache_hit else 'No'} | {r.batch_size} | {r.total_cost:.4f} |")
    lines.extend([
        "",
        "---",
        "",
        "## Recommendations",
        "",
        "1. **Keep batch window at 2s** for latency/cost balance; increase to 3–5s if throughput is bursty.",
        "2. **Monitor cache hit rate**; consider semantic cache for similar (not only exact) queries.",
        "3. **Tune routing rules** if too many complex queries are sent to 4B (quality drop) or simple to 12B (wasted cost).",
        "4. **Scale cost constants** (4B vs 12B $/token) to your actual provider for accurate savings.",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    # 20 queries: all arrive within 2s → 1 batch (single embed call)
    queries_20 = list(TEST_QUERIES)
    # Optional: 5 repeats to show cache (25 total, 5 cache hits)
    queries_with_repeats = queries_20 + queries_20[:5]

    embed_fn = make_embed_fn()

    # Baseline: 20 queries, 1 embed call each, all 12B
    baseline_cost, baseline_records = run_baseline_no_optimization(
        queries_20, embed_fn=embed_fn
    )
    num_embed_baseline = len(queries_20)

    # Optimized: 20 queries, 2s window → 1 batch, routing + cache (no repeats in first run)
    pipeline = BatchedCostPipeline(
        batch_window_seconds=BATCH_WINDOW_SECONDS,
        cache_limit=500,
        embed_fn=embed_fn,
    )
    pipeline.process_queries(queries_20, arrival_times=None)
    optimized_cost = pipeline.get_total_cost()
    optimized_records = pipeline.get_records()
    num_embed_optimized = pipeline.optimizer.costs_tracked["embedding_api_calls"]

    print("Batched cost optimization test (20 queries)")
    print("=" * 50)
    print(f"Baseline:  ${baseline_cost:.4f}  ({num_embed_baseline} embed calls, all 12B)")
    print(f"Optimized: ${optimized_cost:.4f}  ({num_embed_optimized} embed calls, routing + batch)")
    print(f"Savings: ${baseline_cost - optimized_cost:.4f}  ({(1 - optimized_cost/baseline_cost)*100:.1f}%)")
    print()

    # With repeats: 25 queries in 2 time windows so last 5 hit cache
    arrival_times_25 = [0.0] * 20 + [2.5] * 5  # first 20 in [0,2), next 5 in [2.5,4.5)
    pipeline2 = BatchedCostPipeline(batch_window_seconds=BATCH_WINDOW_SECONDS, cache_limit=500, embed_fn=embed_fn)
    pipeline2.process_queries(queries_with_repeats, arrival_times=arrival_times_25)
    opt_with_cache = pipeline2.get_total_cost()
    records_with_cache = pipeline2.get_records()
    cache_hits = sum(1 for r in records_with_cache if r.cache_hit)
    print(f"With 5 repeated queries in 2 windows (25 total): {cache_hits} cache hits, total ${opt_with_cache:.4f}")
    print()

    out_path = Path("cost_optimization_results.md")
    generate_report_md(
        baseline_cost=baseline_cost,
        optimized_cost=optimized_cost,
        baseline_records=baseline_records,
        optimized_records=optimized_records,
        num_embed_calls_baseline=num_embed_baseline,
        num_embed_calls_optimized=num_embed_optimized,
        out_path=out_path,
    )
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
