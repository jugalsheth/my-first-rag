# Cost Optimization Results

## Simple English: What This Does

**The problem:** Answering 20 user questions the old way cost more because we (1) called the embedding API 20 times (once per question) and (2) used the big, expensive model for every question.

**What we did:**

1. **Batch embeddings.** We wait up to 2 seconds and collect questions. Then we send them to the embedding API in one go instead of 20 separate calls. So 20 questions → 1 API call. That cuts embedding cost a lot (about 95% fewer calls).

2. **Use the right model for each question.** Simple questions like “What is RAG?” go to a small, cheap model (4B). Harder questions like “Compare X and Y” go to the smarter, more expensive model (12B). So we spend less on easy questions and still get good answers on hard ones.

3. **Cache repeats.** If the same question is asked again, we return the cached answer. No extra embedding call and no extra model call, so cost is zero for that repeat.

**Result:** For 20 questions, cost went down by about **42%** (from $0.0029 to $0.0017 in our test). The more questions we batch and the more repeats we cache, the more we save.

---

## Summary

| Metric | Before (baseline) | After (optimized) |
|--------|--------------------|--------------------|
| Total cost | $0.0029 | $0.0017 |
| Embedding API calls | 20 | 1 |
| **Savings** | **$0.0012** | **41.9%** |

**Optimizations applied:**
- **Batching:** Queries queued for 2 seconds, then batch embedded (single API call per window).
- **Embedding call reduction:** 20 → 1 calls (95% reduction).
- **Model routing:** Simple queries → Gemma 4B (cheap), complex → Gemma 12B (smart).
- **Cache:** Repeated queries served from cache (0 embed + 0 LLM cost).

---

## Cost by query type (optimized)

| Type | Count | Total cost ($) | Avg cost/query ($) |
|------|-------|----------------|---------------------|
| complex_miss | 10 | 0.0013 | 0.000127 |
| simple_miss | 10 | 0.0004 | 0.000042 |

---

## Per-query detail (optimized)

| # | Query | Model | Cache hit | Batch size | Total cost ($) |
|---|-------|-------|-----------|------------|----------------|
| 1 | What is RAG? | gemma_4b | No | 20 | 0.0000 |
| 2 | Who wrote the CRAG paper? | gemma_4b | No | 20 | 0.0000 |
| 3 | When was BERT released? | gemma_4b | No | 20 | 0.0000 |
| 4 | What is chunk size? | gemma_4b | No | 20 | 0.0000 |
| 5 | Who invented the transformer? | gemma_4b | No | 20 | 0.0000 |
| 6 | When to use dense retrieval? | gemma_4b | No | 20 | 0.0000 |
| 7 | What is embedding? | gemma_4b | No | 20 | 0.0000 |
| 8 | Which model is best for RAG? | gemma_4b | No | 20 | 0.0000 |
| 9 | What is top-k? | gemma_4b | No | 20 | 0.0000 |
| 10 | Who proposed HyDE? | gemma_4b | No | 20 | 0.0000 |
| 11 | Compare dense and sparse retrieval for RAG syste… | gemma_12b | No | 20 | 0.0001 |
| 12 | Analyze the trade-offs between chunk size and re… | gemma_12b | No | 20 | 0.0001 |
| 13 | Explain how self-RAG improves answer quality thr… | gemma_12b | No | 20 | 0.0001 |
| 14 | Why does multi-query retrieval improve coverage? | gemma_12b | No | 20 | 0.0001 |
| 15 | How does re-ranking improve precision in two-sta… | gemma_12b | No | 20 | 0.0001 |
| 16 | Compare CRAG and Self-RAG routing strategies. | gemma_12b | No | 20 | 0.0001 |
| 17 | Explain the role of uncertainty estimation in co… | gemma_12b | No | 20 | 0.0001 |
| 18 | Why is chunk overlap important in document split… | gemma_12b | No | 20 | 0.0001 |
| 19 | How do hypothetical documents bridge the semanti… | gemma_12b | No | 20 | 0.0001 |
| 20 | Analyze when to use local vs web fallback in CRAG. | gemma_12b | No | 20 | 0.0001 |

---

## Recommendations

1. **Keep batch window at 2s** for latency/cost balance; increase to 3–5s if throughput is bursty.
2. **Monitor cache hit rate**; consider semantic cache for similar (not only exact) queries.
3. **Tune routing rules** if too many complex queries are sent to 4B (quality drop) or simple to 12B (wasted cost).
4. **Scale cost constants** (4B vs 12B $/token) to your actual provider for accurate savings.
