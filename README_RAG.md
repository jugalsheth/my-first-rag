# Simple RAG System - Quick Start

## Setup (Run Tonight)

1. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```
   (On macOS, use `pip3` instead of `pip`)

2. **That's it!** Everything is ready to go.

## Run Tomorrow Morning

### Standard Mode
Simply execute:
```bash
python3 rag_system.py
```

The system will:
- Load the embedding model (first run may take a minute)
- Load `sample_document.txt` into ChromaDB
- Run example queries
- Enter interactive mode for you to ask questions

### Benchmark Mode
To compare three embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2, BAAI/bge-small-en-v1.5):
```bash
python3 rag_system.py benchmark
```

This will:
- Test each model with 3 specific questions about RAG
- Create separate ChromaDB collections for each model
- Print retrieved chunks for easy comparison
- Clean up temporary collections after completion

### Chunking Experiment
To test different chunking strategies (Small: 256 tokens, Medium: 512 tokens, Large: 1024 tokens):
```bash
python3 chunk_experiment.py
```

This will:
- Test 3 chunking strategies using RecursiveCharacterTextSplitter from LangChain
- Index each strategy into separate ChromaDB collections
- Query all three with the same question for side-by-side comparison
- Show similarity scores, context length, and overlap analysis
- Help you understand the trade-offs between chunk sizes

### RAGAS Evaluation (Production Scoring)
To automatically score and compare chunking strategies using RAGAS metrics:
```bash
# Basic mode (template-based answers)
python3 rag_evaluator.py

# With Google Gemini for better answer generation (recommended)
# Option 1: Set environment variable directly
export GEMINI_API_KEY=your_gemini_key_here
python3 rag_evaluator.py --gemini

# Option 2: Use .env file (recommended - more secure)
# Create a .env file with: GEMINI_API_KEY=your_key_here
python3 rag_evaluator.py --gemini
```

**Prerequisites:**
1. Run `chunk_experiment.py` first to create the collections
2. For best results, set `GEMINI_API_KEY` environment variable

**Setting up API Key Securely:**
```bash
# Create a .env file (this file is gitignored and won't be committed)
echo "GEMINI_API_KEY=your_actual_key_here" > .env

# Or set it as an environment variable
export GEMINI_API_KEY=your_actual_key_here
```

**⚠️ Security Note:** Never commit API keys to git! The `.env` file is automatically ignored.

**What it evaluates:**
- **Faithfulness**: Does the answer come from retrieved context? (Anti-hallucination)
- **Answer Relevancy**: Does the answer actually answer the question?
- **Context Precision**: Are the retrieved chunks actually relevant?
- **Context Recall**: Does the context contain all needed information?

**Output:**
- Comprehensive comparison table showing which strategy wins
- Winners by metric (the "smoking gun" analysis)
- Production recommendation based on data
- Natural language insights explaining the results

### Multi-Query RAG (Query Expansion)
To test query expansion using Gemini to generate multiple query variations:

```bash
# Basic usage (uses default question and collection)
python3 multi_query_rag.py

# Custom question
python3 multi_query_rag.py "What are the trade-offs of using small vs large chunks in RAG?"

# Use a different collection (e.g., medium or large chunks)
python3 multi_query_rag.py --collection chunk_experiment_medium "Your question here"
```

**Prerequisites:**
1. Run `chunk_experiment.py` first to create the collections
2. Set `GEMINI_API_KEY` environment variable (required for query expansion)

**What it does:**
1. Takes your original question
2. Uses Gemini to generate 3 query variations (different phrasings/angles)
3. Searches ChromaDB with all 3 queries + original
4. Combines and deduplicates results
5. Compares to single-query baseline
6. Calculates coverage improvement percentage

**Output:**
- Generated query variations (see how Gemini rephrased your question)
- Coverage analysis: How many unique chunks did multi-query find vs single query?
- Coverage improvement percentage
- Side-by-side chunk comparison (shows which chunks are "new")
- Query-by-query breakdown (what each variation found)
- Token cost trade-off analysis (3x queries = 3x API calls)

**Key Metrics:**
- **Coverage Improvement**: `(Multi-Query Chunks - Single Query Chunks) / Single Query Chunks * 100%`
- **New Chunks Found**: Unique chunks that only multi-query discovered
- **Overlap Analysis**: Which chunks were found by both approaches

**Use Cases:**
- Testing if query expansion improves retrieval coverage
- Understanding how different phrasings find different content
- Evaluating the cost/benefit trade-off of multi-query approaches
- Research: Data for your Medium article on query expansion techniques

### HyDE RAG (Hypothetical Document Embeddings)
To test HyDE (generate a hypothetical answer, embed it, retrieve, and compare vs standard retrieval):

```bash
# A/B test (3 default questions)
python3 hyde_rag.py

# Single custom question (recommended starting point)
python3 hyde_rag.py --single "Explain the benefits of query rewriting in Advanced RAG"

# Choose a specific collection + top-k
python3 hyde_rag.py --collection chunk_experiment_small --topk 3
```

**Prerequisites:**
1. Run `chunk_experiment.py` first to create the collections
2. Set `GEMINI_API_KEY` (recommended). If not set, HyDE uses a template hypothetical answer (still works, less powerful).

**What it shows:**
- The generated hypothetical answer (HyDE)
- Top-k chunks retrieved by HyDE vs standard
- Which method found the "golden chunk" (highest relevance vs the original question)

### Self-RAG (Retrieval Grading)
To test Self-RAG - system can say "I don't know":

```bash
# Run test suite (3 questions: good match, partial match, bad match)
python3 self_rag.py

# Single question
python3 self_rag.py --single "What are the 3 types of RAG?"

# Custom threshold (2.0 = loose, 3.0 = balanced, 4.0 = strict)
python3 self_rag.py --threshold 4.0
```

**Prerequisites:**
1. Run `chunk_experiment.py` first to create the collections
2. Set `GEMINI_API_KEY` environment variable (required for relevance judging)

**What it does:**
1. Retrieves top-k chunks from ChromaDB
2. Uses Gemini as JUDGE to score relevance (1-5 scale) for each chunk
3. Calculates average relevance score
4. Only answers if average >= threshold, otherwise declines

**Output:**
- Retrieved chunks with relevance scores (1-5)
- Average relevance score
- System decision (answer or decline)
- Answer (if threshold met) or "I don't have enough information"

**Important Discovery - Multi-Layer Protection:**
During testing, we discovered the system has **two layers of protection**:
1. **Retrieval Grading Layer**: Gemini judges chunk relevance (1-5 scale) before answering
2. **Answer Generation Layer**: Even if threshold is met, the answer generator can still decline

**Test Results Example:**
- Question: "What are the 3 types of RAG?" (expected to answer)
- With threshold 3.0: Declined (avg score 2.0) ✓ Correctly conservative
- With threshold 2.0: Attempted to answer, but answer generator said "I don't have enough information" ✓
- **Finding**: The chunks retrieved didn't actually contain the answer, proving both layers work correctly

**Threshold Recommendations:**
- **Threshold = 2.0**: Too loose (may hallucinate)
- **Threshold = 3.0**: Balanced (recommended default)
- **Threshold = 4.0**: Too strict (may miss valid answers)

### CRAG (Corrective RAG with Web Search)
To test CRAG - intelligent routing to web search when local knowledge is insufficient:

```bash
# Run test suite (3 questions: good local, bad local, hybrid)
python3 crag_system.py

# Single question
python3 crag_system.py --single "What are the 3 types of RAG?"

# Choose collection
python3 crag_system.py --collection chunk_experiment_medium
```

**Prerequisites:**
1. Run `chunk_experiment.py` first to create the collections
2. Set `GEMINI_API_KEY` environment variable (required for relevance judging)
3. Set `TAVILY_API_KEY` environment variable (required for web search)
   - Get free API key at: https://tavily.com (1000 requests/month free)

**What it does:**
1. Retrieves top-k chunks from local ChromaDB
2. Uses Gemini to score relevance (1-5 scale) for each chunk
3. Routes based on average relevance score:
   - **4.0-5.0**: Use local docs only (high confidence)
   - **3.0-3.9**: Hybrid mode (combine local + web)
   - **2.0-2.9**: Use web search only (low confidence)
   - **0.0-1.9**: Decline to answer ("I don't know")
4. Generates answer from selected source(s)

**Output:**
- Retrieved chunks with relevance scores (1-5)
- Average relevance score
- Source decision (local/web/hybrid/decline)
- Web search results (if applicable)
- Final answer

**Routing Decision Matrix:**
| Relevance Score | Action | Cost |
|----------------|--------|------|
| 4.0-5.0 | Local only | 1 embedding + 1 LLM call |
| 3.0-3.9 | Hybrid (Local + Web) | 1 embedding + 1 web search + 1 LLM call |
| 2.0-2.9 | Web only | 1 embedding + 1 web search + 1 LLM call |
| 0.0-1.9 | Decline | 1 embedding call only |

**Key Discovery:**
- CRAG extends Self-RAG's "I don't know" with "I'll search the web"
- Hybrid mode provides best of both worlds: authoritative local knowledge + current web information
- Web search is valuable for time-sensitive queries and questions outside local knowledge base

## Files Created

- `requirements.txt` - All dependencies
- `sample_document.txt` - Sample RAG content to query
- `rag_system.py` - Complete RAG system implementation
- `chunk_experiment.py` - Chunking strategy comparison tool
- `rag_evaluator.py` - RAGAS-based evaluation system for production scoring
- `hyde_rag.py` - HyDE (Hypothetical Document Embeddings) experiment
- `multi_query_rag.py` - Multi-query RAG with query expansion comparison
- `self_rag.py` - Self-RAG with retrieval grading
- `crag_system.py` - CRAG (Corrective RAG) with web search fallback
- `chroma_db/` - Vector database (created automatically)

## Customization

- **Add your own documents:** Modify `sample_document.txt` or add more files
- **Change embedding model:** Edit the model name in `rag_system.py` (line 28)
- **Add LLM:** Enhance the `query()` method to use OpenAI, Anthropic, or a local model for better answers

## Research Workflow

1. **Setup**: Run `chunk_experiment.py` to create chunking strategy collections
2. **Evaluate**: Run `rag_evaluator.py` to score each strategy with RAGAS metrics
3. **Query Expansion**: Run `multi_query_rag.py` to test multi-query retrieval
4. **Web Search Fallback**: Run `crag_system.py` to test intelligent routing to web search
5. **Decide**: Use the data-driven recommendations to choose production settings

## Security

- **API Keys**: Never commit API keys to the repository
- **`.env` file**: Create a `.env` file for local API keys (automatically gitignored)
- **Environment Variables**: Use environment variables for API keys in production
- **Git History**: If you accidentally committed an API key, rotate it immediately and remove it from git history

## Notes

- ChromaDB stores data locally in `./chroma_db/`
- First run downloads the embedding model (~90MB)
- RAGAS evaluation works without API keys (uses template answers)
- For best evaluation results, use Google Gemini API (set `GEMINI_API_KEY` in `.env` file or as environment variable)

