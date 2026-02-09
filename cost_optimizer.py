"""
Day 18/90: RAG Cost Optimization System
Reduces production costs by 70%+ through:
- Batch embedding generation (fewer API calls)
- Smart cache eviction (cost-based, LRU)
- Embedding compression (float32 → int8)
- Context trimming (token optimization)
- Cost tracking and ROI analysis
"""

from __future__ import annotations

import time
import hashlib
import json
from collections import OrderedDict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Default cost assumptions (per unit) - override for your provider
DEFAULT_COSTS = {
    "embedding_per_1k_tokens": 0.0001,
    "embedding_per_call_overhead": 0.00002,
    "llm_input_per_1k_tokens": 0.002,
    "llm_output_per_1k_tokens": 0.006,
    "storage_per_1k_bytes_per_month": 0.00001,
}


class CostOptimizer:
    """
    Cost-aware RAG optimizer: batch embeddings, cost-based cache eviction,
    embedding compression, and context trimming with full cost tracking.
    """

    def __init__(
        self,
        cache_limit: int = 1000,
        batch_size: int = 10,
        max_context_tokens: int = 400,
        costs: Optional[Dict[str, float]] = None,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
        tokenizer_fn: Optional[Callable[[str], int]] = None,
    ):
        """
        Args:
            cache_limit: Max cached (query_embedding, answer, chunks) entries.
            batch_size: Queries per batch for embedding API (e.g. 10 → 1 call).
            max_context_tokens: Trim context to this many tokens before LLM.
            costs: Override DEFAULT_COSTS for your provider.
            embed_fn: (queries: List[str]) -> List[List[float]]. If None, batch_embed uses placeholder.
            tokenizer_fn: (text: str) -> token_count. If None, uses char-based estimate (~4 chars/token).
        """
        self.cache_limit = cache_limit
        self.batch_size = batch_size
        self.max_context_tokens = max_context_tokens
        self.costs = {**DEFAULT_COSTS, **(costs or {})}
        self.embed_fn = embed_fn
        self.tokenizer_fn = tokenizer_fn or self._approx_tokens

        # Batch queue for batching: list of (query, callback_or_None)
        self.batch_queue: List[Tuple[str, Optional[Any]]] = []

        # Cost-aware cache: key = query_hash, value = {
        #   "query", "embedding" (decompressed), "embedding_compressed", "scale",
        #   "answer", "chunks", "access_count", "last_used", "generation_cost"
        # }
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # Running cost totals (in dollars)
        self.costs_tracked = {
            "embedding": 0.0,
            "embedding_api_calls": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "llm": 0.0,
            "storage_bytes": 0,
            "storage": 0.0,
        }

        # Baseline comparison (set by test harness)
        self.baseline_costs: Optional[Dict[str, float]] = None

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """Rough token count (~4 chars per token for English)."""
        return max(1, len(text) // 4)

    def _hash_query(self, query: str) -> str:
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:32]

    # ------------------------- Batch embeddings -------------------------

    def batch_embed(self, queries: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Embed multiple queries in batches (e.g. 10 queries = 1 API call).
        Tracks embedding cost and API call count.
        """
        batch_size = batch_size or self.batch_size
        all_embeddings: List[List[float]] = []
        total_tokens = 0
        num_calls = 0

        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            num_calls += 1
            for q in batch:
                total_tokens += self._approx_tokens(q)

            if self.embed_fn:
                batch_embs = self.embed_fn(batch)
                for emb in batch_embs:
                    if isinstance(emb, np.ndarray):
                        all_embeddings.append(emb.tolist())
                    else:
                        all_embeddings.append(list(emb))
            else:
                # Placeholder: zero vectors of dim 384 (MiniLM-like) for testing
                dim = 384
                for _ in batch:
                    all_embeddings.append([0.0] * dim)

            # Cost: per-call overhead + per-token
            self.costs_tracked["embedding_api_calls"] += 1
            self.costs_tracked["embedding"] += self.costs["embedding_per_call_overhead"]
            self.costs_tracked["embedding"] += (total_tokens / 1000.0) * self.costs["embedding_per_1k_tokens"]
            total_tokens = 0

        return all_embeddings

    # ------------------------- Embedding compression -------------------------

    def compress_embeddings(
        self, embeddings: List[List[float]]
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Quantize float32 → int8 per vector. Returns list of (int8_array, scale).
        Original 768d float32 = 3072 bytes → 768d int8 + 4 byte scale ≈ 772 bytes (~75% savings).
        """
        result = []
        for emb in embeddings:
            arr = np.array(emb, dtype=np.float32)
            scale = float(np.max(np.abs(arr)))
            if scale < 1e-9:
                scale = 1.0
            int8 = np.clip(np.round(arr / scale * 127), -128, 127).astype(np.int8)
            result.append((int8, scale))
        return result

    def decompress_embedding(self, compressed: Tuple[np.ndarray, float]) -> np.ndarray:
        """int8 + scale → float32."""
        int8, scale = compressed
        return (int8.astype(np.float32) / 127.0) * scale

    def compressed_embedding_bytes(self, dim: int) -> int:
        """Bytes per compressed embedding: dim + 4 for scale."""
        return dim + 4

    def original_embedding_bytes(self, dim: int) -> int:
        """Bytes per float32 embedding."""
        return dim * 4

    # ------------------------- Smart cache eviction -------------------------

    def _eviction_score(self, entry: Dict[str, Any]) -> float:
        """
        Higher = keep. Evict low-value: rarely used, cheap to regenerate.
        score = access_count / (generation_cost + epsilon)
        """
        cost = max(entry.get("generation_cost", 0.001), 1e-6)
        return entry.get("access_count", 0) / cost

    def smart_evict(self, num_to_evict: int = 1) -> None:
        """
        Evict entries with lowest value (access_count / generation_cost).
        Keeps frequently accessed and expensive-to-regenerate items.
        """
        if len(self.cache) < self.cache_limit or num_to_evict <= 0:
            return
        # Build list of (key, score), sort by score ascending, evict lowest
        items = [
            (k, self._eviction_score(v))
            for k, v in self.cache.items()
        ]
        items.sort(key=lambda x: x[1])
        for i in range(min(num_to_evict, len(items))):
            key = items[i][0]
            if key in self.cache:
                del self.cache[key]

    def get_from_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Check cache; return entry and update access/last_used. Move to end for LRU."""
        key = self._hash_query(query)
        if key not in self.cache:
            return None
        entry = self.cache[key]
        entry["access_count"] = entry.get("access_count", 0) + 1
        entry["last_used"] = time.time()
        self.cache.move_to_end(key)
        return entry

    def put_in_cache(
        self,
        query: str,
        embedding: List[float],
        answer: str,
        chunks: List[str],
        generation_cost: float,
    ) -> None:
        """Store compressed embedding and metadata. Evict if over limit."""
        key = self._hash_query(query)
        compressed_list = self.compress_embeddings([embedding])
        comp, scale = compressed_list[0]
        dim = len(embedding)
        storage_delta = self.compressed_embedding_bytes(dim)
        self.costs_tracked["storage_bytes"] += storage_delta
        self.costs_tracked["storage"] += (storage_delta / 1000.0) * self.costs["storage_per_1k_bytes_per_month"]

        entry = {
            "query": query,
            "embedding": embedding,
            "embedding_compressed": (comp, scale),
            "answer": answer,
            "chunks": chunks,
            "access_count": 0,
            "last_used": time.time(),
            "generation_cost": generation_cost,
        }
        while len(self.cache) >= self.cache_limit:
            self.smart_evict(1)
        self.cache[key] = entry
        self.cache.move_to_end(key)

    # ------------------------- Context optimization -------------------------

    def optimize_context(self, chunks: List[str]) -> List[str]:
        """
        Trim retrieved chunks to stay within max_context_tokens.
        Keeps most relevant (first) chunks and truncates last if needed.
        """
        if not chunks:
            return []
        budget = self.max_context_tokens
        result = []
        for c in chunks:
            tokens = self.tokenizer_fn(c)
            if budget <= 0:
                break
            if tokens <= budget:
                result.append(c)
                budget -= tokens
            else:
                # Truncate to fit: keep first N chars ~= budget tokens
                keep_chars = budget * 4
                result.append(c[:keep_chars].rsplit(" ", 1)[0] if keep_chars < len(c) else c[:keep_chars])
                budget = 0
        return result

    def context_tokens_before_after(self, chunks: List[str]) -> Tuple[int, int]:
        """Return (tokens_before, tokens_after) for reporting."""
        before = sum(self.tokenizer_fn(c) for c in chunks)
        after_chunks = self.optimize_context(chunks)
        after = sum(self.tokenizer_fn(c) for c in after_chunks)
        return before, after

    # ------------------------- Cost and ROI -------------------------

    def add_llm_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record LLM token usage and cost."""
        self.costs_tracked["llm_input_tokens"] += input_tokens
        self.costs_tracked["llm_output_tokens"] += output_tokens
        self.costs_tracked["llm"] += (input_tokens / 1000.0) * self.costs["llm_input_per_1k_tokens"]
        self.costs_tracked["llm"] += (output_tokens / 1000.0) * self.costs["llm_output_per_1k_tokens"]

    def get_savings_report(self) -> Dict[str, Any]:
        """
        Return cost breakdown, savings vs baseline, and ROI metrics.
        """
        total = (
            self.costs_tracked["embedding"]
            + self.costs_tracked["llm"]
            + self.costs_tracked["storage"]
        )
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "costs": {
                "embedding": round(self.costs_tracked["embedding"], 4),
                "embedding_api_calls": self.costs_tracked["embedding_api_calls"],
                "llm": round(self.costs_tracked["llm"], 4),
                "llm_input_tokens": self.costs_tracked["llm_input_tokens"],
                "llm_output_tokens": self.costs_tracked["llm_output_tokens"],
                "storage": round(self.costs_tracked["storage"], 4),
                "storage_bytes": self.costs_tracked["storage_bytes"],
                "total": round(total, 4),
            },
            "cache": {
                "size": len(self.cache),
                "limit": self.cache_limit,
            },
        }
        if self.baseline_costs:
            base_total = self.baseline_costs.get("total", 0) or sum(self.baseline_costs.values())
            savings_pct = (1 - total / base_total) * 100 if base_total > 0 else 0
            report["baseline"] = self.baseline_costs
            report["savings"] = {
                "total_baseline": round(base_total, 4),
                "total_optimized": round(total, 4),
                "savings_percent": round(savings_pct, 1),
                "savings_dollars": round(base_total - total, 4),
            }
        return report

    def get_roi_analysis(self, queries_processed: int) -> Dict[str, Any]:
        """ROI: savings per 1K queries, monthly (30K), annual."""
        report = self.get_savings_report()
        if "savings" not in report or queries_processed <= 0:
            return {}
        saved = report["savings"]["savings_dollars"]
        per_1k = saved * (1000.0 / queries_processed)
        monthly_30k = per_1k * 30
        annual = monthly_30k * 12
        return {
            "queries_processed": queries_processed,
            "savings_per_1000_queries": round(per_1k, 2),
            "monthly_30k_queries": round(monthly_30k, 2),
            "annual": round(annual, 2),
        }


def create_optimizer_with_sentence_transformer(
    model_name: str = "all-MiniLM-L6-v2",
    cache_limit: int = 1000,
    batch_size: int = 10,
    max_context_tokens: int = 400,
) -> CostOptimizer:
    """Build a CostOptimizer that uses sentence-transformers for real embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)

        def embed_fn(queries: List[str]) -> List[List[float]]:
            embs = model.encode(queries, convert_to_numpy=True)
            return [embs[i].tolist() for i in range(len(queries))]

        return CostOptimizer(
            cache_limit=cache_limit,
            batch_size=batch_size,
            max_context_tokens=max_context_tokens,
            embed_fn=embed_fn,
        )
    except Exception as e:
        raise RuntimeError(f"Could not load SentenceTransformer: {e}") from e
