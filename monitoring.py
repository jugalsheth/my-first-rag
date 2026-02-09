"""
Day 17: RAG Monitoring & Observability
Tracks production metrics: latency (p50, p95, p99), cache hit rate, cost, quality (RAGAS), errors.
Dashboard-ready JSON export for visualization.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _percentile(sorted_arr: List[float], p: float) -> float:
    """Compute percentile (0-100). Returns 0.0 if empty."""
    if not sorted_arr:
        return 0.0
    k = (len(sorted_arr) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_arr) else f
    return sorted_arr[f] + (k - f) * (sorted_arr[c] - sorted_arr[f])


class RAGMonitor:
    """
    Production RAG metrics: latency percentiles, cache performance, cost, quality, errors.
    Use log_query() after each RAG call; get_summary() / export_dashboard() for reporting.
    """

    def __init__(
        self,
        max_query_history: int = 10_000,
        metrics_file: Optional[str] = None,
    ):
        self.max_query_history = max_query_history
        self.metrics_file = Path(metrics_file) if metrics_file else None

        self._queries: deque = deque(maxlen=max_query_history)
        self._latencies_ms: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_cost_usd = 0.0
        self._errors: List[Dict[str, Any]] = []
        self._quality_scores: List[float] = []

    def log_query(
        self,
        query: str,
        latency_ms: float,
        cached: bool,
        cost_usd: float = 0.0,
        quality_score: Optional[float] = None,
        error: Optional[str] = None,
        cache_tier: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log one RAG query for observability.

        Args:
            query: User query text.
            latency_ms: End-to-end latency in milliseconds.
            cached: True if answer came from cache (any tier).
            cost_usd: Cost of this query in USD (e.g. API cost).
            quality_score: Optional RAGAS or 1-5 quality score.
            error: Optional error message (e.g. "rate_limit", "timeout").
            cache_tier: "exact" | "semantic" | None (full RAG).
            metadata: Optional extra fields for dashboard.
        """
        if cached:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        self._latencies_ms.append(latency_ms)
        self._total_cost_usd += cost_usd
        if quality_score is not None:
            self._quality_scores.append(quality_score)

        if error:
            self._errors.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query[:200],
                "error": error,
                "latency_ms": latency_ms,
                "cached": cached,
            })

        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query[:500],
            "latency_ms": round(latency_ms, 2),
            "cached": cached,
            "cost_usd": round(cost_usd, 6),
            "cache_tier": cache_tier,
        }
        if quality_score is not None:
            entry["quality_score"] = round(quality_score, 4)
        if error:
            entry["error"] = error
        if metadata:
            entry["metadata"] = metadata

        self._queries.append(entry)

    def get_summary(self) -> Dict[str, Any]:
        """
        Return summary statistics for dashboards and alerts.
        Includes p50, p95, p99 latency, cache hit rate, total cost, error count.
        """
        n = len(self._latencies_ms)
        sorted_lat = sorted(self._latencies_ms) if self._latencies_ms else []

        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests) if total_requests else 0.0

        avg_quality = (
            sum(self._quality_scores) / len(self._quality_scores)
            if self._quality_scores else None
        )

        return {
            "total_queries": n,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": round(hit_rate, 4),
            "avg_latency_ms": round(sum(self._latencies_ms) / n, 2) if n else 0.0,
            "p50_latency_ms": round(_percentile(sorted_lat, 50), 2),
            "p95_latency_ms": round(_percentile(sorted_lat, 95), 2),
            "p99_latency_ms": round(_percentile(sorted_lat, 99), 2),
            "total_cost_usd": round(self._total_cost_usd, 6),
            "avg_cost_per_query_usd": round(self._total_cost_usd / n, 6) if n else 0.0,
            "error_count": len(self._errors),
            "avg_quality_score": round(avg_quality, 4) if avg_quality is not None else None,
            "queries_with_quality": len(self._quality_scores),
        }

    def get_timeseries(self, max_points: int = 100) -> List[Dict[str, Any]]:
        """Last N query entries for time-series charts (e.g. latency over time)."""
        queries = list(self._queries)
        if len(queries) <= max_points:
            return queries
        return queries[-max_points:]

    def export_dashboard(
        self,
        filepath: Optional[str] = None,
        include_timeseries: bool = True,
        max_timeseries: int = 100,
    ) -> Dict[str, Any]:
        """
        Export dashboard-ready JSON. Optionally write to file.

        Returns:
            Dict with summary, timeseries, and recent_errors for visualization.
        """
        filepath = Path(filepath) if filepath else self.metrics_file
        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "summary": self.get_summary(),
            "recent_errors": self._errors[-50:],
        }
        if include_timeseries:
            payload["timeseries"] = self.get_timeseries(max_points=max_timeseries)

        if filepath:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(payload, f, indent=2)
        return payload

    def reset(self) -> None:
        """Clear all metrics (e.g. for a new test run)."""
        self._queries.clear()
        self._latencies_ms.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_cost_usd = 0.0
        self._errors.clear()
        self._quality_scores.clear()
