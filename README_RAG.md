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

## Files Created

- `requirements.txt` - All dependencies
- `sample_document.txt` - Sample RAG content to query
- `rag_system.py` - Complete RAG system implementation
- `chunk_experiment.py` - Chunking strategy comparison tool
- `rag_evaluator.py` - RAGAS-based evaluation system for production scoring
- `multi_query_rag.py` - Multi-query RAG with query expansion comparison
- `chroma_db/` - Vector database (created automatically)

## Customization

- **Add your own documents:** Modify `sample_document.txt` or add more files
- **Change embedding model:** Edit the model name in `rag_system.py` (line 28)
- **Add LLM:** Enhance the `query()` method to use OpenAI, Anthropic, or a local model for better answers

## Research Workflow

1. **Setup**: Run `chunk_experiment.py` to create chunking strategy collections
2. **Evaluate**: Run `rag_evaluator.py` to score each strategy with RAGAS metrics
3. **Query Expansion**: Run `multi_query_rag.py` to test multi-query retrieval
4. **Decide**: Use the data-driven recommendations to choose production settings

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

