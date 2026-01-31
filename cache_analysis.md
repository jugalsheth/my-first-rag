# Cache Performance Analysis

**Generated:** 2026-01-31T13:12:04.637361

## Summary

- **Total Queries:** 20
- **Repeat Rate:** 50.0%
- **Cache Hit Rate:** 50.0%
- **Total Time Reduction:** 73.5%
- **Average Latency Reduction:** 73.8%
- **Cost Reduction:** 50.0%

## Performance Comparison

| Metric | Uncached | Cached | Improvement |
|--------|----------|--------|-------------|
| Total Time | 917.19 ms | 243.03 ms | 73.5% faster |
| Average Latency | 45.60 ms | 11.93 ms | 73.8% faster |
| Full RAG Calls | 20 | 10 | 50.0% reduction |

## Cache Breakdown

| Tier | Hits | Hit Rate | Avg Latency |
|------|------|----------|-------------|
| Exact Match | 10 | 50.0% | 0.00 ms |
| Semantic Similarity | 0 | 0.0% | 0.00 ms |
| Full RAG | 10 | 50.0% | 15.64 ms |

## Cost Analysis

- **Uncached Cost:** $0.0020
- **Cached Cost:** $0.0010
- **Cost Savings:** $0.0010
- **Cost Reduction:** 50.0%

