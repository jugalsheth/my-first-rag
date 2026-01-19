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

## Files Created

- `requirements.txt` - All dependencies
- `sample_document.txt` - Sample RAG content to query
- `rag_system.py` - Complete RAG system implementation
- `chunk_experiment.py` - Chunking strategy comparison tool
- `chroma_db/` - Vector database (created automatically)

## Customization

- **Add your own documents:** Modify `sample_document.txt` or add more files
- **Change embedding model:** Edit the model name in `rag_system.py` (line 28)
- **Add LLM:** Enhance the `query()` method to use OpenAI, Anthropic, or a local model for better answers

## Notes

- ChromaDB stores data locally in `./chroma_db/`
- First run downloads the embedding model (~90MB)
- All processing happens locally - no API keys needed!

