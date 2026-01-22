# RAG Research Log

This document tracks research findings and discoveries as we build and test our RAG system.

## Day 1: Initial RAG System Setup

**Date:** 2026-01-13

**What we built:**
- Basic RAG system using ChromaDB and sentence-transformers
- Simple document loading and querying
- Sample document with RAG information

**Key Findings:**
- ChromaDB works well for local vector storage (no API needed)
- Sentence-transformers provides free, local embeddings
- Simple RAG setup can be done in <100 lines of code

**Files Created:**
- `rag_system.py` - Core RAG implementation
- `sample_document.txt` - Test corpus

---

## Day 2: Embedding Model Benchmark

**Date:** 2026-01-13

**What we tested:**
- Compared three embedding models:
  - `all-MiniLM-L6-v2` (384 dimensions, fast)
  - `all-mpnet-base-v2` (768 dimensions, higher quality)
  - `BAAI/bge-small-en-v1.5` (384 dimensions, SOTA)

**Key Findings:**
- **MPNet (768 dim) beat MiniLM (384 dim) at scale** - Larger dimension models capture more semantic nuance
- Embedding model choice significantly impacts retrieval quality
- Trade-off between speed (MiniLM) and quality (MPNet)

**Discovery:**
- Higher-dimensional embeddings (768 vs 384) provide better semantic matching for complex queries
- Model choice depends on speed vs quality requirements

**Files Created/Modified:**
- Enhanced `rag_system.py` with benchmark mode
- `requirements.txt` - Added all dependencies

---

## Day 3: Chunking Strategy Experiment

**Date:** 2026-01-13

**What we tested:**
- Three chunking strategies:
  - **Small**: 256 tokens, 20 token overlap
  - **Medium**: 512 tokens, 50 token overlap
  - **Large**: 1024 tokens, 100 token overlap

**Key Findings:**
- **The "Laser vs. Floodlight" Discovery:**
  - **Small chunks (256)**: Higher similarity scores but may miss context (Laser - precise but narrow)
  - **Large chunks (1024)**: Better context but potentially more noise (Floodlight - broad but may include irrelevant info)
  - **Medium chunks (512)**: Balanced approach for general use

**Metrics Analyzed:**
- Token distribution statistics
- Coverage ratio (how much of document is captured)
- Overlap effectiveness (context preservation across boundaries)
- Keyword density
- Information density

**Visualizations:**
- `chunk_token_distribution.png` - Histogram showing token distribution across strategies

**Discovery:**
- Chunk size has a direct trade-off: precision vs completeness
- Overlap is critical for preserving context when concepts span chunk boundaries
- Optimal chunk size depends on query type (factoid vs conceptual)

**Files Created:**
- `chunk_experiment.py` - Comprehensive chunking analysis tool
- `chunk_token_distribution.png` - Visualization

---

## Day 4-7: RAGAS Evaluation

**Date:** 2026-01-13

**What we tested:**
- Production-grade evaluation using RAGAS framework
- Four metrics:
  - **Faithfulness**: Prevents hallucinations
  - **Answer Relevancy**: Question-answer alignment
  - **Context Precision**: Retrieval quality (not computed due to InstructorLLM limitation)
  - **Context Recall**: Information completeness (not computed due to InstructorLLM limitation)

**Key Findings:**
- **The RAGAS Verdict:**
  - **Small (256) chunks actually produced more 'Faithful' and 'Relevant' answers**
  - This contradicts the assumption that larger chunks = better answers
  - Higher precision (Small) can lead to better faithfulness scores

**Technical Challenges Solved:**
- RAGAS 0.4+ requires InstructorLLM (not compatible with Gemini)
- Implemented custom evaluation metrics using Gemini directly
- Added rate limiting for Gemini API (5-10 RPM limits)
- Implemented fallbacks for API quota exhaustion

**Rate Limiting Discoveries:**
- Free tier Gemini: 5-10 RPM, 20 requests/day
- Gemma models have separate quota (30 RPM)
- Need strategic delays and retry logic

**Files Created:**
- `rag_evaluator.py` - RAGAS evaluation system
- Custom evaluation wrapper for Gemini compatibility

---

## Day 8: Multi-Query RAG (Query Expansion)

**Date:** 2026-01-13

**What we built:**
- Multi-query RAG system using Gemini for query expansion
- Generates 3 query variations from original question
- Combines and deduplicates results for better coverage

**Key Features:**
- **Query Expansion**: Uses Gemini to generate alternative phrasings
- **Multi-Query Retrieval**: Searches ChromaDB with all queries
- **Deduplication**: Combines results, keeps highest similarity scores
- **Coverage Analysis**: Calculates improvement vs single-query baseline

**Key Metrics:**
- **Coverage Improvement**: `(Multi-Query Chunks - Single Query Chunks) / Single Query Chunks * 100%`
- **New Chunks Found**: Unique chunks only discovered by multi-query
- **Overlap Analysis**: Which chunks both approaches found

**Expected Insights (To Be Tested):**
- Does multi-query find new relevant chunks or just more noise?
- What's the token cost trade-off? (3x queries = 3x API calls)
- When is query expansion worth the extra cost?

**Files Created:**
- `multi_query_rag.py` - Multi-query RAG with query expansion

**Research Questions for Testing:**
1. What coverage improvement does multi-query achieve?
2. Do query variations find genuinely different relevant chunks?
3. Is the 3x API cost worth the coverage improvement?
4. Which query types benefit most from expansion?

---

## Day 9: HyDE RAG (Hypothetical Document Embeddings)

**Date:** 2026-01-20

**What we built:**
- HyDE experiment runner using your Day 3 ChromaDB collection
- A/B comparison:
  - **Standard retrieval**: embed the question → retrieve
  - **HyDE retrieval**: generate a ~200-word hypothetical answer with Gemini → embed it → retrieve

**What we show (per question):**
- The generated hypothetical answer
- Top-3 chunks retrieved by **HyDE**
- Top-3 chunks retrieved by **Standard**
- A simple “golden chunk” heuristic:
  - Score each retrieved chunk by cosine similarity to the original question (embedding space)
  - Golden chunk = highest-scoring chunk across Standard ∪ HyDE
  - Report whether Standard and/or HyDE retrieved it

**A/B Test Questions:**
1. Technical: “Explain the benefits of query rewriting”
2. Conceptual: “Why does HyDE improve retrieval?”
3. Comparison: “What’s the difference between HyDE and standard retrieval?”

**Cost / Latency Analysis (conceptual):**
- Standard: 1 embedding call
- HyDE: 1 LLM call + 1 embedding call
- Trade-off: potential retrieval improvement vs higher cost/latency

**Files Created:**
- `hyde_rag.py` - HyDE vs Standard retrieval experiment runner

---

## Day 9: HyDE RAG (Hypothetical Document Embeddings)

**Date:** 2026-01-20

**What we built:**
- HyDE system that generates hypothetical answers before retrieval
- Uses Gemini to create ~200-word hypothetical answer from question
- Embeds the hypothetical answer (not the question) for retrieval
- Compares HyDE retrieval vs standard query embedding

**Key Features:**
- **Hypothetical Answer Generation**: Gemini creates a plausible answer first
- **Answer-Based Retrieval**: Embeds the hypothetical answer instead of the question
- **A/B Comparison**: Side-by-side comparison with standard retrieval
- **Golden Chunk Detection**: Identifies most relevant chunk and which method found it

**Test Questions:**
- Technical: "Explain the benefits of query rewriting"
- Conceptual: "Why does HyDE improve retrieval?"
- Comparison: "What's the difference between HyDE and standard retrieval?"

**Key Metrics:**
- Chunks retrieved by HyDE vs Standard
- Relevance scores for each chunk
- Which approach found the "golden chunk" (most relevant)

**Cost Analysis:**
- Standard: 1 embedding call
- HyDE: 1 LLM call (generate fake answer) + 1 embedding call
- Extra cost: ~$0.0001 per query (with Gemini)
- Trade-off: Better retrieval vs. higher cost/latency

**Files Created:**
- `hyde_rag.py` - HyDE vs Standard retrieval comparison

---

## Day 10: Self-RAG (Retrieval Grading)

**Date:** 2026-01-20

**What we built:**
- Self-RAG system with retrieval grading
- Gemini acts as a JUDGE to score chunk relevance (1-5 scale)
- System only answers if average relevance >= threshold
- Can decline to answer: "I don't have enough information"

**Key Features:**
- **Retrieval Grading**: Gemini judges relevance of each chunk (1-5 scale)
- **Threshold-Based Decision**: Only answers if average score meets threshold
- **Graceful Decline**: Says "I don't know" when retrieval quality is poor
- **Configurable Threshold**: Experiment with 2.0, 3.0, 4.0 to find optimal balance

**Test Questions:**
- **Good match**: "What are the 3 types of RAG?" (should answer)
- **Partial match**: "How does blockchain improve RAG?" (should decline)
- **Bad match**: "What's the weather today?" (should decline)

**Key Metrics:**
- Relevance scores (1-5) for each retrieved chunk
- Average relevance score
- System decision (answer or decline)
- Threshold tuning results

**Threshold Tuning Findings:**
- **Threshold = 2.0**: Answers everything (too loose, may hallucinate)
- **Threshold = 3.0**: Balanced (recommended default)
- **Threshold = 4.0**: Only answers perfect matches (too strict, may miss valid answers)

**The Goal:**
Prove the system can say "I don't know" when it doesn't have enough information, preventing hallucinations.

**Files Created:**
- `self_rag.py` - Self-RAG with retrieval grading

---

## Research Methodology

### Evaluation Framework

We use a data-driven approach:

1. **Hypothesis Formation**: Based on research questions
2. **Experimental Design**: Systematic testing with controlled variables
3. **Data Collection**: Automated metrics and evaluations
4. **Analysis**: Statistical comparison and interpretation
5. **Documentation**: Clear findings and recommendations

### Tools & Technologies

- **ChromaDB**: Local vector database
- **Sentence Transformers**: Embedding models
- **LangChain**: Document processing
- **RAGAS**: Evaluation framework
- **Google Gemini**: LLM for answer generation and query expansion
- **Rich**: Beautiful terminal output
- **Matplotlib**: Visualizations

### Key Principles

1. **Reproducibility**: All experiments are scripted and repeatable
2. **Data-Driven**: Decisions based on metrics, not intuition
3. **Production-Ready**: Tests use real-world constraints (rate limits, API quotas)
4. **Transparency**: All findings documented with data

---

## Ongoing Research Questions

### Current Questions

1. **Multi-Query Effectiveness**: Does query expansion improve retrieval enough to justify 3x cost?
2. **Optimal Query Variations**: How many variations is optimal?
3. **Query Type Dependency**: Which queries benefit most from expansion?

### Future Research Directions

- [ ] Semantic chunking (topic-based rather than token-based)
- [ ] Adaptive chunk sizing based on query type
- [ ] Re-ranking impact on retrieval quality
- [ ] Cross-lingual chunking strategies
- [ ] Hybrid dense-sparse retrieval
- [ ] Real-time chunk optimization
- [ ] Query routing strategies

---

## Key Discoveries Summary

### 1. Embedding Dimension Impact
- **Finding**: 768-dim models (MPNet) outperform 384-dim models (MiniLM) at scale
- **Implication**: Higher dimensions capture more semantic nuance

### 2. Chunking Paradox (Laser vs. Floodlight)
- **Finding**: Small chunks have higher similarity but large chunks have better context
- **Implication**: Trade-off between precision and completeness

### 3. RAGAS Verdict
- **Finding**: Small (256) chunks produced more faithful and relevant answers
- **Implication**: Higher precision can lead to better faithfulness scores

### 4. Query Expansion (Day 8 - In Progress)
- **Hypothesis**: Multi-query improves coverage by finding chunks missed by single queries
- **Testing**: Coverage improvement percentage and cost/benefit analysis

---

## Production Recommendations

Based on current research:

### Recommended Baseline (2026)

- **Chunking Strategy**: Small (256 tokens, 20 overlap) OR Medium (512 tokens, 50 overlap)
  - Use Small for factoid queries or when precision is critical
  - Use Medium for balanced performance
- **Embedding Model**: `all-mpnet-base-v2` (768 dimensions) for quality, `all-MiniLM-L6-v2` for speed
- **Evaluation**: Use RAGAS metrics (Faithfulness + Answer Relevancy) for continuous monitoring
- **Query Expansion**: Test multi-query for your specific use case (Day 8 research)

### Key Takeaways

1. **Small chunks can be better**: Don't assume bigger = better
2. **Metrics matter**: Use RAGAS, not just similarity scores
3. **Query type matters**: Factoid vs conceptual queries need different strategies
4. **Test systematically**: Data-driven decisions beat intuition

---

## Contributing to Research

When adding new findings:

1. Document in this file with date
2. Include key findings with data
3. Note any challenges solved
4. Update recommendations if findings change them
5. Link to relevant code files

**Format:**
```
## Day X: [Title]
**Date:** YYYY-MM-DD
**What we tested:**
**Key Findings:**
**Files Created/Modified:**
```

---

*Last Updated: 2026-01-13*
