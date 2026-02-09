# 90-DAY RAG EXPERT PLAN (CURSOR EDITION)

**Start:** Jan 17, 2026 | **Current:** Day 15/90 | **Budget:** $20/month

---

## DAILY WORKFLOW

Each day, tell Cursor:
```
Day X of 90-day RAG plan. Build [FEATURE] from plan below.
Create working code + test + document results.
```

---

## WEEK 1: FOUNDATION ✅ COMPLETE

### Day 1: Basic RAG
- Build: query → embed → search → generate pipeline
- Test: 5 questions on RAG papers
- Metric: Answer quality (1-5 scale)

### Day 2: Embedding Comparison
- Build: Test OpenAI vs sentence-transformers
- Test: 10 queries, compare retrieval quality
- Metric: Precision, recall, cost

### Day 3: Chunk Size Optimization  
- Build: Test 200, 500, 1000 char chunks
- Test: Same 10 queries across sizes
- Metric: Retrieval accuracy, context relevance

### Day 4: RAGAS Evaluation
- Build: Integrate RAGAS metrics (faithfulness, relevance, context precision)
- Test: Evaluate all previous experiments
- Metric: RAGAS scores

### Day 5: Retrieval Optimization
- Build: Re-ranking, query expansion, hybrid search
- Test: Compare to baseline
- Metric: Top-K accuracy improvement

### Day 6: Write Article #1
- Topic: "Week 1 RAG learnings"
- Length: 2,500+ words
- Data: All Week 1 test results

### Day 7: Rest + Review

---

## WEEK 2: ADVANCED TECHNIQUES ✅ COMPLETE

### Day 8: Multi-Query RAG
- Build: Generate 3 query variations → parallel search → merge results
- Test: Coverage comparison (single vs multi)
- Metric: +% chunks found, 3x cost

### Day 9: HyDE (Hypothetical Documents)
- Build: Generate fake answer → embed → search with fake answer
- Test: Semantic gap bridging
- Metric: Retrieval relevance improvement

### Day 10: Self-RAG (Quality Gating)
- Build: Grade chunks (1-5) → threshold 3.0 → answer or decline
- Test: 20 questions, 3 thresholds (2.0, 3.0, 4.0)
- Metric: Answer rate vs accuracy trade-off

### Day 11: CRAG (Corrective RAG)
- Build: Score local → route (local/hybrid/web) → Tavily API fallback
- Test: 3 routing strategies
- Metric: 3-tier beats binary (67% vs 33%)

### Day 12: Agentic RAG
- Build: Generate → grade → refine query → retry (max 3)
- Test: Vague questions, track iterations
- Metric: Score improvement per attempt

### Day 13: Newsletter #2
- Topic: "5 advanced RAG techniques compared"
- Platform: Substack

### Day 14: Article #2
- Topic: "Advanced RAG in production"
- Length: 3,500+ words

---

## WEEK 3: PRODUCTION (CURRENT)

### Day 15: Caching ← YOU ARE HERE
**Build:**
```python
class CachedRAG:
    def __init__(self):
        self.exact_cache = {}  # hash -> answer
        self.semantic_cache = []  # [(embedding, answer)]
        self.ttl = 3600  # 1 hour
        
    def query(self, text):
        # Tier 1: Exact match
        hash_key = hashlib.md5(text.encode()).hexdigest()
        if hash_key in self.exact_cache:
            return self.exact_cache[hash_key]
        
        # Tier 2: Semantic similarity
        query_emb = embed(text)
        for cached_emb, cached_ans in self.semantic_cache:
            if cosine_sim(query_emb, cached_emb) > 0.95:
                return cached_ans
        
        # Tier 3: Full RAG
        answer = full_rag_pipeline(text)
        self.exact_cache[hash_key] = answer
        self.semantic_cache.append((query_emb, answer))
        return answer
```

**Test:** 20 queries (10 unique, 10 repeats)
**Metric:** Hit rate, latency reduction, cost savings

### Day 16: A/B Testing
**Build:** Test harness to compare RAG variants
```python
def ab_test(variant_a, variant_b, test_queries):
    results = {'a': [], 'b': []}
    for q in test_queries:
        results['a'].append(variant_a.eval(q))
        results['b'].append(variant_b.eval(q))
    return statistical_test(results)
```

**Test:** Multi-Query vs Standard, HyDE vs Standard
**Metric:** Win rate, statistical significance (p < 0.05)

### Day 17: Monitoring
**Build:** Logging, metrics dashboard, alerts
```python
class MonitoredRAG:
    def query(self, text):
        start = time.time()
        try:
            result = self.rag.query(text)
            log_success(text, result, time.time()-start)
            return result
        except Exception as e:
            log_error(text, e)
            alert_on_call()
            return fallback_response()
```

**Test:** Simulate 100 queries, track all metrics
**Metric:** Error rate, p95 latency, throughput

### Day 18: Cost Optimization
**Build:** Batch embeddings, smart caching, cheaper models for simple queries
```python
def route_by_complexity(query):
    if is_simple(query):
        return fast_cheap_model(query)  # Gemma 4B
    else:
        return powerful_model(query)  # Gemma 12B
```

**Test:** Measure cost before/after optimizations
**Metric:** $/query reduction

### Day 19: Error Handling
**Build:** Retry with backoff, circuit breaker, graceful degradation
```python
@retry(max_attempts=3, backoff=exponential)
def robust_rag_query(text):
    if circuit_breaker.is_open():
        return cached_fallback()
    try:
        return rag.query(text)
    except RateLimitError:
        circuit_breaker.open()
        return "System busy, try again"
```

**Test:** Simulate API failures, rate limits
**Metric:** Success rate under failure conditions

### Day 20: Newsletter #3
**Topic:** "Production RAG: Performance, reliability, cost"

### Day 21: Rest + Week 3 Review

---

## WEEK 4: EVALUATION & TESTING

### Day 22: Automated Testing Suite
**Build:** Pytest suite with 50+ test cases covering all RAG variants
**Metric:** Code coverage, pass rate

### Day 23: Benchmark Dataset Creation
**Build:** Curate 100 Q&A pairs with ground truth from RAG papers
**Metric:** Dataset quality (manual review)

### Day 24: Cross-Validation Framework
**Build:** K-fold validation for RAG hyperparameters
**Metric:** Optimal chunk size, top-k, threshold per use case

### Day 25: Adversarial Testing
**Build:** Test edge cases (empty docs, malformed queries, very long context)
**Metric:** Failure rate, error messages quality

### Day 26: Regression Testing
**Build:** Automated checks that new features don't break old ones
**Metric:** All tests pass on every commit

### Day 27: Newsletter #4
**Topic:** "How to evaluate RAG properly"

### Day 28: Rest + Week 4 Review

---

## WEEK 5: SCALABILITY

### Day 29: Async RAG
**Build:** Convert to async/await for concurrent queries
**Metric:** Throughput improvement (queries/sec)

### Day 30: Batch Processing
**Build:** Process multiple queries in single API call
**Metric:** Cost reduction via batching

### Day 31: Connection Pooling
**Build:** Reuse DB connections, HTTP sessions
**Metric:** Latency reduction from connection reuse

### Day 32: Load Testing
**Build:** Simulate 1000 concurrent users with Locust
**Metric:** System behavior under load

### Day 33: Horizontal Scaling
**Build:** Multi-instance RAG with load balancer
**Metric:** Linear scalability proof

### Day 34: Article #3
**Topic:** "Scaling RAG to production traffic"

### Day 35: Rest + Month 1 Review

---

## WEEK 6: ADVANCED RETRIEVAL

### Day 36: Dense + Sparse Hybrid
**Build:** Combine semantic (dense) + keyword (BM25 sparse) search
**Metric:** Recall improvement

### Day 37: Cross-Encoder Re-Ranking
**Build:** Use BERT cross-encoder to re-rank top-K results
**Metric:** Precision@3 improvement

### Day 38: Query Decomposition
**Build:** Break complex questions into sub-queries
**Metric:** Multi-hop question accuracy

### Day 39: Contextual Compression
**Build:** Extract only relevant sentences from retrieved chunks
**Metric:** Token usage reduction without accuracy loss

### Day 40: Retrieval Fusion
**Build:** Combine multiple retrieval strategies (reciprocal rank fusion)
**Metric:** Best-of-all-worlds performance

### Day 41: Newsletter #5
**Topic:** "Advanced retrieval techniques"

### Day 42: Rest + Week 6 Review

---

## WEEK 7: GENERATION QUALITY

### Day 43: Prompt Engineering
**Build:** Test 10 prompt templates for generation
**Metric:** RAGAS faithfulness score per template

### Day 44: Few-Shot Examples
**Build:** Add 3-5 examples to prompt for consistency
**Metric:** Output format compliance rate

### Day 45: Citation Generation
**Build:** Generate answers with source citations [1], [2]
**Metric:** Citation accuracy (correct source attribution)

### Day 46: Hallucination Detection
**Build:** Detect when LLM adds info not in context
**Metric:** False positive rate on hallucination detection

### Day 47: Answer Confidence Scoring
**Build:** Score answer certainty (low/medium/high)
**Metric:** Calibration (confidence matches actual accuracy)

### Day 48: Newsletter #6
**Topic:** "Improving generation quality"

### Day 49: Rest + Week 7 Review

---

## WEEK 8: DOMAIN ADAPTATION

### Day 50: Domain-Specific Embeddings
**Build:** Fine-tune embeddings on domain data
**Metric:** Retrieval improvement on domain queries

### Day 51: Custom Chunking Strategy
**Build:** Semantic chunking (split on topics, not chars)
**Metric:** Context relevance improvement

### Day 52: Metadata Filtering
**Build:** Filter by document type, date, author before retrieval
**Metric:** Precision improvement via pre-filtering

### Day 53: Multi-Modal RAG
**Build:** Retrieve from text + images + tables
**Metric:** Coverage on multi-modal questions

### Day 54: Knowledge Graph Integration
**Build:** Combine vector search with graph traversal
**Metric:** Relational query accuracy

### Day 55: Article #4
**Topic:** "Domain-specific RAG optimization"

### Day 56: Rest + Week 8 Review

---

## WEEK 9: AGENTIC RAG

### Day 57: Tool-Using RAG
**Build:** Let RAG call external APIs (calculator, weather, DB)
**Metric:** Task success rate with tools

### Day 58: Multi-Step Reasoning
**Build:** Chain-of-thought prompting for complex queries
**Metric:** Accuracy on multi-step problems

### Day 59: Self-Correction
**Build:** RAG verifies its own answer, retries if wrong
**Metric:** Accuracy improvement from self-correction

### Day 60: Memory & State
**Build:** Maintain conversation history across queries
**Metric:** Context retention over conversation

### Day 61: Planning & Execution
**Build:** RAG makes plan (steps) then executes each
**Metric:** Complex task completion rate

### Day 62: Newsletter #7
**Topic:** "Agentic RAG patterns"

### Day 63: Rest + Month 2 Review

---

## WEEK 10: SPECIALIZED TECHNIQUES

### Day 64: Temporal RAG
**Build:** Handle time-dependent queries ("What was X in 2020?")
**Metric:** Temporal accuracy

### Day 65: Conversational RAG
**Build:** Multi-turn dialogue with context tracking
**Metric:** Coherence over conversation

### Day 66: Summarization RAG
**Build:** Retrieve many docs, summarize into single answer
**Metric:** Summary quality (ROUGE scores)

### Day 67: Fact Verification RAG
**Build:** Check claims against knowledge base
**Metric:** Fact-check accuracy (true/false classification)

### Day 68: Comparative RAG
**Build:** Compare entities ("X vs Y")
**Metric:** Completeness of comparison

### Day 69: Newsletter #8
**Topic:** "Specialized RAG applications"

### Day 70: Rest + Week 10 Review

---

## WEEK 11: DEPLOYMENT

### Day 71: FastAPI Service
**Build:** REST API for RAG (POST /query endpoint)
**Metric:** API response time

### Day 72: Docker Containerization
**Build:** Dockerfile + docker-compose for RAG stack
**Metric:** Reproducible deployment

### Day 73: Cloud Deployment (AWS/GCP)
**Build:** Deploy to cloud, configure autoscaling
**Metric:** Production uptime

### Day 74: Rate Limiting & Auth
**Build:** API keys, rate limits per user
**Metric:** Security & fair usage enforcement

### Day 75: Streaming Responses
**Build:** Stream tokens as generated (SSE/WebSocket)
**Metric:** Time to first token

### Day 76: Article #5
**Topic:** "Deploying production RAG"

### Day 77: Rest + Week 11 Review

---

## WEEK 12: ADVANCED TOPICS

### Day 78: Private RAG (Local Models)
**Build:** Run entirely locally (Ollama + local embeddings)
**Metric:** Zero external API calls

### Day 79: Multi-Tenant RAG
**Build:** Isolated knowledge bases per user/org
**Metric:** Data isolation guarantees

### Day 80: RAG Security
**Build:** Prompt injection defense, output sanitization
**Metric:** Attack resistance

### Day 81: RAG Analytics
**Build:** Usage dashboards, popular queries, feedback loop
**Metric:** User insights extracted

### Day 82: RAG Fine-Tuning
**Build:** Fine-tune LLM on domain QA pairs
**Metric:** Performance improvement from fine-tuning

### Day 83: Newsletter #9
**Topic:** "Advanced RAG topics"

### Day 84: Rest + Month 3 Review

---

## WEEK 13: SYNTHESIS & FUTURE

### Day 85: RAG Comparison Report
**Build:** Compare all 20+ techniques built, create decision matrix
**Metric:** Comprehensive performance table

### Day 86: Open Source Contribution
**Build:** Clean up best code, publish package to PyPI
**Metric:** Package downloads, GitHub stars

### Day 87: Research Paper Review
**Build:** Write academic-style paper on findings
**Metric:** Novel insights documented

### Day 88: Future Roadmap
**Build:** Identify 10 unsolved RAG problems, propose solutions
**Metric:** Research agenda clarity

### Day 89: Article #6 (Final)
**Topic:** "90 days of RAG: What I learned"

### Day 90: Celebration + Next Steps
**Review:** All metrics, code, content
**Share:** Full journey summary
**Plan:** What's next (job search, startup, research?)

---

## KEY METRICS TO TRACK DAILY

**Technical:**
- Retrieval accuracy (precision, recall)
- Generation quality (RAGAS scores)
- Latency (p50, p95, p99)
- Cost ($/query)
- Error rate (%)

**Content:**
- Posts published (2/day)
- Newsletter subscribers
- Article views
- GitHub stars
- Engagement rate

---

## TESTING COMMANDS FOR CURSOR

Day 15 example:
```
Build Day 15 caching system from plan:
- CachedRAG class with 3 tiers
- Test with 20 queries
- Measure hit rate, latency, cost
- Output results to cache_results.md
```

Day 20 example:
```
Write Newsletter #3 from Week 3 work:
- Synthesis of Days 15-19
- Production readiness theme
- Include all metrics from tests
- 1,200 words, conversational tone
```

---

## RESEARCH PAPERS TO REFERENCE

1. RAG Survey (2024) - arxiv.org/abs/2312.10997
2. Self-RAG - arxiv.org/abs/2310.11511
3. CRAG - arxiv.org/abs/2401.15884
4. HyDE - arxiv.org/abs/2212.10496
5. RAGAS - arxiv.org/abs/2309.15217

Add all to Cursor context for reference.

---

## SUCCESS CRITERIA (Day 90)

✅ 20+ RAG techniques implemented & tested
✅ Production-ready system (<100ms, 99.9% uptime)
✅ 90 LinkedIn posts, 90 tweets, 12 newsletters, 6 articles
✅ 1,000+ LinkedIn followers, 500+ Twitter, 100+ newsletter subs
✅ GitHub repo with 50+ stars
✅ Total cost <$20/month

---

## CURSOR USAGE TIP

Each day:
```
Day X/90. Build [FEATURE] from CURSOR_90_DAY_PLAN.md.
Write working code, test it, document results.
```

Cursor will handle everything with this plan + papers in context.
