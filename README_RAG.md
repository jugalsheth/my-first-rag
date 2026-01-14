# Simple RAG System - Quick Start

## Setup (Run Tonight)

1. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```
   (On macOS, use `pip3` instead of `pip`)

2. **That's it!** Everything is ready to go.

## Run Tomorrow Morning

Simply execute:
```bash
python3 rag_system.py
```

The system will:
- Load the embedding model (first run may take a minute)
- Load `sample_document.txt` into ChromaDB
- Run example queries
- Enter interactive mode for you to ask questions

## Files Created

- `requirements.txt` - All dependencies
- `sample_document.txt` - Sample RAG content to query
- `rag_system.py` - Complete RAG system implementation
- `chroma_db/` - Vector database (created automatically)

## Customization

- **Add your own documents:** Modify `sample_document.txt` or add more files
- **Change embedding model:** Edit the model name in `rag_system.py` (line 28)
- **Add LLM:** Enhance the `query()` method to use OpenAI, Anthropic, or a local model for better answers

## Notes

- ChromaDB stores data locally in `./chroma_db/`
- First run downloads the embedding model (~90MB)
- All processing happens locally - no API keys needed!

