# Day 17: RAG Monitoring & Observability

## Metrics Tracked

- **Latency:** p50, p95, p99 (ms)
- **Cache:** hit rate, hits vs misses
- **Cost:** total USD, avg per query
- **Quality:** optional RAGAS/quality score over time
- **Errors:** count + recent errors with context (query, timestamp, error type)

## Usage

```python
from monitoring import RAGMonitor

monitor = RAGMonitor()

# After each RAG call:
monitor.log_query(
    query="What is RAG?",
    latency_ms=450.0,
    cached=False,
    cost_usd=0.0003,
    quality_score=0.85,
    error=None,
    cache_tier=None,
)

# Summary for alerts/dashboards
summary = monitor.get_summary()
# -> p50_latency_ms, p95_latency_ms, cache_hit_rate, total_cost_usd, error_count, ...

# Export JSON for visualization
monitor.export_dashboard("metrics_dashboard.json")
```

## Integration with Cached RAG

Wrap your CachedRAG (or any RAG) and log each query:

```python
from monitoring import RAGMonitor
from cached_rag import CachedRAG

monitor = RAGMonitor()
rag = CachedRAG(...)

def query_with_monitoring(q: str):
    start = time.perf_counter()
    try:
        answer, chunks, tier = rag.query(q)  # use your actual API
        latency_ms = (time.perf_counter() - start) * 1000
        cached = tier in ("exact", "semantic")
        monitor.log_query(q, latency_ms, cached, cost_usd=0.0002, cache_tier=tier)
        return answer
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        monitor.log_query(q, latency_ms, False, 0.0, error=str(e))
        raise
```

## Test Run

```bash
python test_monitoring.py
```

- Runs 50 simulated queries (mix cached/uncached, 3 failures).
- Prints summary and writes `metrics_dashboard.json`.

## Results (example)

- **Avg latency:** ~450 ms (acceptable)
- **Cache hit rate:** ~68% (saves ~68% of API calls)
- **Cost:** ~$0.003/query (under budget)
- **Dashboard:** `metrics_dashboard.json` with `summary`, `timeseries`, `recent_errors`
