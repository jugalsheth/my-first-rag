"""
Batched cost optimization pipeline:
- Queue queries for 2 seconds, batch embed (single API call per window)
- Per-query tracking: model used (4B/12B), cache hit, batch size, total cost
- Before vs after: baseline (all 12B, 1 embed call per query) vs optimized (routing + batch + cache)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from query_complexity import classify_query, route_model
from cost_optimizer import CostOptimizer, DEFAULT_COSTS
from cost_routing_rag import DEFAULT_COST_4B, DEFAULT_COST_12B

# Default: queue window 2 seconds
BATCH_WINDOW_SECONDS = 2.0


@dataclass
class PerQueryRecord:
    """Per-query cost and metadata."""
    query: str
    model_used: str  # "gemma_4b" | "gemma_12b"
    cache_hit: bool
    batch_size: int  # number of queries in the embedding batch (0 if cache hit)
    embed_cost: float
    llm_cost: float
    total_cost: float
    complexity: str = ""
    original_index: int = -1


def _group_by_time_window(
    queries: List[str],
    arrival_times: Optional[List[float]] = None,
    window_seconds: float = BATCH_WINDOW_SECONDS,
) -> List[List[Tuple[str, int]]]:
    """
    Group queries into batches by time window. Each batch = queries in [t, t+window).
    Returns list of batches; each batch is list of (query, original_index).
    If arrival_times is None, treat all as t=0 (one batch).
    """
    if not queries:
        return []
    if arrival_times is None:
        return [[(q, i) for i, q in enumerate(queries)]]
    if len(arrival_times) != len(queries):
        arrival_times = [0.0] * len(queries)
    indexed = list(zip(queries, arrival_times, range(len(queries))))
    indexed.sort(key=lambda x: x[1])
    batches: List[List[Tuple[str, int]]] = []
    current_start = indexed[0][1]
    current: List[Tuple[str, int]] = []
    for q, t, i in indexed:
        if t >= current_start + window_seconds and current:
            batches.append(current)
            current = []
            current_start = t
        current.append((q, i))
    if current:
        batches.append(current)
    return batches


def _simulate_llm_tokens(query: str, chunks: List[str], max_context: int = 1200, max_output: int = 200) -> Tuple[int, int]:
    in_tok = min(max(1, (len("\n\n".join(chunks)) + len(query) + 50) // 4), max_context)
    out_tok = max(50, min(max_output, len(query) // 2))
    return in_tok, out_tok


def _llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    costs = DEFAULT_COST_4B if model == "gemma_4b" else DEFAULT_COST_12B
    return (input_tokens / 1000.0) * costs["input_per_1k_tokens"] + (
        output_tokens / 1000.0
    ) * costs["output_per_1k_tokens"]


class BatchedCostPipeline:
    """
    Pipeline: 2s batch window → batch embed → cache check → route (4B/12B) → per-query cost.
    """

    def __init__(
        self,
        batch_window_seconds: float = BATCH_WINDOW_SECONDS,
        cache_limit: int = 500,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        costs: Optional[Dict[str, float]] = None,
    ):
        self.batch_window_seconds = batch_window_seconds
        self.optimizer = CostOptimizer(
            cache_limit=cache_limit,
            batch_size=999,  # we control batching by time window
            costs={**DEFAULT_COSTS, **(costs or {})},
            embed_fn=embed_fn,
        )
        self.per_query_records: List[PerQueryRecord] = []
        self._embed_cost_before_batch: float = 0.0

    def process_queries(
        self,
        queries: List[str],
        arrival_times: Optional[List[float]] = None,
        chunks_per_query: int = 3,
        ctx_tokens: int = 400,
    ) -> List[PerQueryRecord]:
        """
        Process queries in 2s windows: batch embed per window, then for each query
        check cache, route to 4B/12B, record model_used, cache_hit, batch_size, total_cost.
        """
        self.per_query_records = []
        self._embed_cost_before_batch = self.optimizer.costs_tracked["embedding"]
        # Match baseline: same context size as run_baseline_no_optimization
        fake_chunks = ["x" * (ctx_tokens // 6)] * chunks_per_query

        batches = _group_by_time_window(queries, arrival_times, self.batch_window_seconds)

        for batch_with_idx in batches:
            batch_queries = [q for q, _ in batch_with_idx]
            need_embed: List[Tuple[str, int]] = []
            for q, idx in batch_with_idx:
                hit = self.optimizer.get_from_cache(q)
                if hit:
                    self.per_query_records.append(
                        PerQueryRecord(
                            query=q,
                            model_used="gemma_12b",  # cached answer from any model
                            cache_hit=True,
                            batch_size=0,
                            embed_cost=0.0,
                            llm_cost=0.0,
                            total_cost=0.0,
                            complexity=classify_query(q),
                            original_index=idx,
                        )
                    )
                else:
                    need_embed.append((q, idx))

            if not need_embed:
                continue

            # Batch embed all that need embedding (single API call)
            need_embed_queries = [q for q, _ in need_embed]
            embeddings = self.optimizer.batch_embed(need_embed_queries, batch_size=len(need_embed_queries))
            embed_cost_this_call = (
                self.optimizer.costs_tracked["embedding"] - self._embed_cost_before_batch
            )
            self._embed_cost_before_batch = self.optimizer.costs_tracked["embedding"]
            # Allocate embed cost evenly across this batch
            n = len(need_embed_queries)
            alloc = embed_cost_this_call / n if n else 0.0

            for (q, idx), emb in zip(need_embed, embeddings):
                model = route_model(q)
                complexity = classify_query(q)
                in_tok, out_tok = _simulate_llm_tokens(q, fake_chunks)
                llm_cost = _llm_cost(model, in_tok, out_tok)
                self.optimizer.add_llm_usage(in_tok, out_tok)
                total = alloc + llm_cost
                self.optimizer.put_in_cache(
                    q, emb, "[answer for " + q[:30] + "]", fake_chunks, generation_cost=total
                )
                self.per_query_records.append(
                    PerQueryRecord(
                        query=q,
                        model_used=model,
                        cache_hit=False,
                        batch_size=n,
                        embed_cost=round(alloc, 6),
                        llm_cost=round(llm_cost, 6),
                        total_cost=round(total, 6),
                        complexity=complexity,
                        original_index=idx,
                    )
                )

        self.per_query_records.sort(key=lambda r: r.original_index)
        return self.per_query_records

    def get_total_cost(self) -> float:
        return sum(r.total_cost for r in self.per_query_records)

    def get_records(self) -> List[PerQueryRecord]:
        return self.per_query_records


def run_baseline_no_optimization(
    queries: List[str],
    embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    chunks_per_query: int = 3,
    ctx_tokens: int = 400,
) -> Tuple[float, List[PerQueryRecord]]:
    """
    Baseline: 1 embedding API call per query, all 12B, no cache.
    Returns (total_cost, list of per-query records).
    """
    records = []
    for i, q in enumerate(queries):
        embed_cost = (
            DEFAULT_COSTS["embedding_per_call_overhead"]
            + (max(1, len(q) // 4) / 1000.0) * DEFAULT_COSTS["embedding_per_1k_tokens"]
        )
        in_tok, out_tok = _simulate_llm_tokens(q, ["x" * (ctx_tokens // 6)] * chunks_per_query)
        llm_cost = _llm_cost("gemma_12b", in_tok, out_tok)
        total = embed_cost + llm_cost
        records.append(
            PerQueryRecord(
                query=q,
                model_used="gemma_12b",
                cache_hit=False,
                batch_size=1,
                embed_cost=round(embed_cost, 6),
                llm_cost=round(llm_cost, 6),
                total_cost=round(total, 6),
                complexity=classify_query(q),
                original_index=i,
            )
        )
    return round(sum(r.total_cost for r in records), 4), records
