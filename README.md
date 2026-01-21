# RAG Chunking Strategy Research Study

🔬 **A comprehensive research study comparing chunking strategies for Retrieval-Augmented Generation (RAG) systems**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Research Overview

This repository contains a systematic investigation into how different chunking strategies impact RAG system performance. We evaluate three chunking approaches (Small, Medium, Large) using production-grade metrics to determine optimal configurations for real-world applications.

### Research Questions

1. **How does chunk size affect retrieval precision and answer quality?**
2. **What is the optimal balance between context completeness and noise reduction?**
3. **Which chunking strategy produces the most faithful and relevant answers?**
4. **How do overlap strategies impact information preservation across chunk boundaries?**

### Key Findings

- **Small chunks (256 tokens)**: High precision, may miss context
- **Medium chunks (512 tokens)**: Balanced precision and completeness
- **Large chunks (1024 tokens)**: Complete context, potential noise

*See evaluation results for data-driven recommendations*

---

## 🎯 Research Methodology

### Experimental Design

1. **Document Preparation**: Standardized test corpus on RAG systems
2. **Chunking Strategies**: Three configurations with varying sizes and overlaps
3. **Embedding Generation**: Consistent embedding model across all strategies
4. **Evaluation Metrics**: RAGAS framework (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
5. **Statistical Analysis**: Comparative analysis with visualizations

### Chunking Configurations

| Strategy | Chunk Size | Overlap | Use Case |
|----------|-----------|---------|----------|
| **Small** | 256 tokens | 20 tokens | Factoid queries, high precision needs |
| **Medium** | 512 tokens | 50 tokens | Balanced queries, general purpose |
| **Large** | 1024 tokens | 100 tokens | Conceptual queries, comprehensive answers |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip3 (Python package manager)
- Google Gemini API key (optional, for enhanced evaluation)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Day3

# Install dependencies
pip3 install -r requirements.txt
```

### Environment Setup

For enhanced evaluation with Google Gemini:

```bash
# Create .env file (automatically gitignored)
echo "GEMINI_API_KEY=your_gemini_key_here" > .env
```

**Get your Gemini API key**: [Google AI Studio](https://makersuite.google.com/app/apikey)

---

## 📊 Running Experiments

### Step 1: Chunking Strategy Experiment

Create and index documents with different chunking strategies:

```bash
python3 chunk_experiment.py
```

**What it does:**
- Splits documents using three chunking strategies
- Indexes each strategy into separate ChromaDB collections
- Generates token distribution visualizations
- Provides statistical analysis and insights

**Output:**
- `chunk_token_distribution.png` - Visual comparison of chunk sizes
- Detailed analysis tables and recommendations
- Natural language insights on chunking trade-offs

### Step 2: RAGAS Evaluation

Evaluate chunking strategies using production metrics:

```bash
# Basic mode (template-based answers)
python3 rag_evaluator.py

# Enhanced mode with Gemini API (recommended)
python3 rag_evaluator.py --gemini
```

**What it evaluates:**
- **Faithfulness**: Prevents hallucinations by ensuring answers come from context
- **Answer Relevancy**: Measures how well answers address the questions
- **Context Precision**: Evaluates relevance of retrieved chunks
- **Context Recall**: Assesses completeness of retrieved information

**Output:**
- Comprehensive comparison table
- Winners by metric analysis
- Production recommendations
- Research insights and interpretations

### Step 3: Embedding Model Benchmark

Compare different embedding models:

```bash
python3 rag_system.py benchmark
```

**Models tested:**
- `all-MiniLM-L6-v2` - Fast, efficient
- `all-mpnet-base-v2` - Higher quality
- `BAAI/bge-small-en-v1.5` - State-of-the-art

### Step 4: Multi-Query RAG (Query Expansion)

Test query expansion using Gemini to generate multiple query variations:

```bash
# Basic usage (uses default question and collection)
python3 multi_query_rag.py

# Custom question
python3 multi_query_rag.py "What are the trade-offs of using small vs large chunks in RAG?"

# Use a different collection
python3 multi_query_rag.py --collection chunk_experiment_medium "Your question here"
```

**What it does:**
- Takes your original question
- Uses Gemini to generate 3 query variations (different phrasings/angles)
- Searches ChromaDB with all 3 queries + original
- Combines and deduplicates results
- Compares to single-query baseline

**Key Metrics:**
- **Coverage Improvement**: Percentage increase in unique chunks found
- **New Chunks Found**: Unique chunks only discovered by multi-query
- **Token Cost Trade-off**: 3x queries = 3x API calls

**Use Cases:**
- Testing if query expansion improves retrieval coverage
- Understanding how different phrasings find different content
- Evaluating cost/benefit trade-off of multi-query approaches

---

### Step 5: HyDE RAG (Hypothetical Document Embeddings)

Test HyDE by generating a hypothetical answer with Gemini, embedding it, and retrieving based on that embedding:

```bash
# Run A/B test (3 questions)
python3 hyde_rag.py

# Run a single custom question
python3 hyde_rag.py --single "Explain the benefits of query rewriting in Advanced RAG"

# Pick a specific collection + top-k
python3 hyde_rag.py --collection chunk_experiment_medium --topk 3
```

**What it does:**
- Standard retrieval: embed the question → retrieve top-k
- HyDE retrieval: generate ~200-word hypothetical answer → embed it → retrieve top-k
- Shows the hypothetical answer + both retrieval sets side-by-side
- Reports which method found the “golden chunk” (highest relevance vs the original question)

**Trade-off:**
- HyDE adds 1 LLM call per question (latency + cost) but can improve recall on hard queries.

---

## 📁 Repository Structure

```
Day3/
├── README.md                 # This file - research overview
├── README_RAG.md            # Quick start guide
├── RESEARCH.md              # Research log and findings
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
│
├── rag_system.py            # Core RAG system implementation
├── chunk_experiment.py      # Chunking strategy comparison
├── rag_evaluator.py         # RAGAS evaluation framework
├── multi_query_rag.py       # Multi-query RAG with query expansion
├── hyde_rag.py              # HyDE vs Standard retrieval experiment
│
├── sample_document.txt      # Test corpus (RAG documentation)
├── chunk_token_distribution.png  # Visualization output
├── multiquery_exhibit.png   # Exhibit: Multi-query experiment diagram (PNG)
├── research_exhibit.png     # Exhibit: Research timeline summary (PNG)
│
├── MEDIUM_ARTICLE.md        # Medium article draft
├── MEDIUM_FORMATTING_GUIDE.md  # Formatting guide for Medium
│
└── chroma_db/              # Vector database (auto-generated)
    └── [collections]       # Indexed document chunks
```

---

## 🔬 Research Components

### 1. RAG System (`rag_system.py`)

Core implementation of a RAG system using:
- **ChromaDB**: Local vector database
- **Sentence Transformers**: Free, local embeddings
- **LangChain**: Document processing and chunking

**Features:**
- Interactive query mode
- Benchmark mode for embedding comparison
- Configurable chunk sizes and overlaps

### 2. Chunking Experiment (`chunk_experiment.py`)

Comprehensive chunking strategy analysis:

**Metrics Calculated:**
- Token distribution statistics
- Coverage ratio analysis
- Overlap effectiveness scores
- Keyword density measurements
- Information density metrics

**Visualizations:**
- Token distribution histograms
- Statistical comparison tables
- Natural language insights

### 3. RAGAS Evaluator (`rag_evaluator.py`)

Production-grade evaluation using RAGAS framework:

**Evaluation Modes:**
- **Template-based**: Works without API keys (baseline)
- **Gemini-enhanced**: Uses Google Gemini for accurate answer generation

**Metrics:**
- Faithfulness (anti-hallucination)
- Answer Relevancy (question-answer alignment)
- Context Precision (retrieval quality)
- Context Recall (information completeness)

### 4. Multi-Query RAG (`multi_query_rag.py`)

Query expansion system for improved retrieval coverage:

**Features:**
- **Query Expansion**: Uses Gemini to generate 3 query variations
- **Multi-Query Retrieval**: Searches ChromaDB with all variations
- **Deduplication**: Combines results, keeps highest similarity
- **Coverage Analysis**: Calculates improvement vs single-query baseline

**Use Cases:**
- Testing query expansion effectiveness
- Understanding retrieval coverage improvements
- Evaluating cost/benefit of multi-query approaches

---

## 📈 Interpreting Results

### Understanding RAGAS Scores

- **0.0 - 0.4**: Poor performance, needs improvement
- **0.4 - 0.7**: Acceptable, room for optimization
- **0.7 - 0.9**: Good performance, production-ready
- **0.9 - 1.0**: Excellent, optimal configuration

### Strategy Selection Guide

**Choose Small (256 tokens) if:**
- You need high precision retrieval
- Queries are factoid (dates, names, specific facts)
- Context window is limited
- Speed is critical

**Choose Medium (512 tokens) if:**
- You need balanced performance
- Queries are mixed (factoid + conceptual)
- General-purpose application
- Good default choice

**Choose Large (1024 tokens) if:**
- You need comprehensive answers
- Queries are conceptual (how/why questions)
- Context completeness is critical
- You can handle potential noise

---

## 🔧 Configuration & Customization

### Customizing Chunking Strategies

Edit `chunk_experiment.py`:

```python
self.strategies = {
    "Small": {"chunk_size": 256, "chunk_overlap": 20},
    "Medium": {"chunk_size": 512, "chunk_overlap": 50},
    "Large": {"chunk_size": 1024, "chunk_overlap": 100},
    # Add your custom strategy here
    "Custom": {"chunk_size": 768, "chunk_overlap": 75}
}
```

### Changing Embedding Models

Edit `rag_system.py` or `rag_evaluator.py`:

```python
# Available models:
# - "all-MiniLM-L6-v2" (fast, 384 dims)
# - "all-mpnet-base-v2" (quality, 768 dims)
# - "BAAI/bge-small-en-v1.5" (SOTA, 384 dims)

embedding_model = SentenceTransformer("your-model-name")
```

### Adding Test Questions

Edit `rag_evaluator.py`:

```python
self.test_questions = [
    "Your custom question 1",
    "Your custom question 2",
    # Add more questions
]
```

---

## 🛠️ Technical Stack

- **Python 3.9+**: Core language
- **ChromaDB 0.4.22**: Vector database
- **Sentence Transformers 2.3.1**: Embedding models
- **LangChain 0.1.0**: Document processing
- **RAGAS 0.4.3**: Evaluation framework
- **Rich 13.7.0**: Beautiful terminal output
- **Matplotlib 3.8.2**: Visualizations
- **Google Gemini API**: Enhanced answer generation (optional)

---

## 📝 Research Workflow

### Complete Research Pipeline

```bash
# 1. Setup
pip3 install -r requirements.txt

# 2. Create chunking strategy collections
python3 chunk_experiment.py

# 3. Evaluate strategies with RAGAS
python3 rag_evaluator.py --gemini

# 4. Compare embedding models (optional)
python3 rag_system.py benchmark

# 5. Test query expansion (optional)
python3 multi_query_rag.py "Your question here"

# 6. Test HyDE (optional)
python3 hyde_rag.py --single "Your question here"

# 7. Analyze results and make recommendations
# (Results displayed in terminal)
```

### Iterative Research Process

1. **Hypothesis**: Formulate research question
2. **Experiment**: Run chunking experiment
3. **Evaluate**: Score with RAGAS metrics
4. **Analyze**: Interpret results
5. **Iterate**: Refine strategies based on findings
6. **Document**: Record insights and recommendations

---

## 🔒 Security & Best Practices

### API Key Management

✅ **DO:**
- Use `.env` file for local development (automatically gitignored)
- Use environment variables in production
- Rotate keys if accidentally exposed
- Use separate keys for development/production

❌ **DON'T:**
- Commit API keys to git
- Share keys in screenshots or documentation
- Use production keys in development

### Git Safety

The repository is configured to ignore:
- `.env` files
- API key files (`*.key`, `*.pem`)
- Database files (`chroma_db/`)
- Python cache (`__pycache__/`)

---

## 🎓 Research Insights

### Key Discoveries

1. **Precision-Completeness Trade-off**: Smaller chunks have higher precision but may miss context; larger chunks provide completeness but introduce noise.

2. **Overlap Effectiveness**: Strategic overlap (10-20% of chunk size) preserves information across boundaries without excessive redundancy.

3. **Query-Type Dependency**: Factoid queries benefit from small chunks; conceptual queries require larger chunks for comprehensive answers.

4. **Embedding Model Impact**: Model choice significantly affects retrieval quality, with larger models generally performing better but slower.

### Future Research Directions

- [ ] Multi-query RAG effectiveness analysis (Day 8 - in progress)
- [ ] Semantic chunking (topic-based)
- [ ] Adaptive chunk sizing based on query type
- [ ] Re-ranking impact on chunk selection
- [ ] Cross-lingual chunking strategies
- [ ] Real-time chunk optimization
- [ ] Query routing strategies

---

## 🤝 Contributing

This is an active research project. Contributions welcome!

### Areas for Contribution

- Additional chunking strategies
- New evaluation metrics
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📚 References & Resources

### Papers & Documentation

- [RAGAS: Evaluation Framework](https://docs.ragas.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

### Related Research

- Retrieval-Augmented Generation (RAG) architectures
- Document chunking strategies
- Vector similarity search optimization
- LLM evaluation metrics

---

## 📊 Results & Visualizations

Results are displayed in the terminal with:
- Rich-formatted tables
- Color-coded metrics
- Statistical summaries
- Natural language insights
- Production recommendations

Visual outputs:
- `chunk_token_distribution.png` - Token distribution comparison

---

## ⚠️ Known Limitations

1. **API Quotas**: Free tier Gemini API has rate limits (5-10 RPM, 20 requests/day)
   - **Solution**: Use Gemma models (30 RPM) or template-based evaluation

2. **RAGAS Compatibility**: RAGAS 0.4+ requires InstructorLLM (not supported by Gemini)
   - **Solution**: Custom evaluation metrics implemented

3. **Model Availability**: Some Gemini models are deprecated
   - **Solution**: Automatic fallback to available models

---

## 📞 Support & Questions

For research questions, issues, or contributions:
- Open an issue on GitHub
- Review existing documentation
- Check the research methodology section

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- ChromaDB team for the excellent vector database
- LangChain for document processing tools
- RAGAS team for evaluation framework
- Sentence Transformers for embedding models
- Google for Gemini API

---

**🔬 Happy Researching!**

*This repository is part of an ongoing research study. Results and methodologies are continuously refined based on new findings and feedback.*
