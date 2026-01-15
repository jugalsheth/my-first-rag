"""
Simple RAG System using ChromaDB and sentence-transformers
Ready to run - just install dependencies and execute!
"""

import os
import warnings

# Suppress harmless urllib3/OpenSSL warning on macOS
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


class SimpleRAG:
    def __init__(self, collection_name: str = "rag_documents", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG system with ChromaDB and sentence-transformers
        
        Args:
            collection_name: Name of the ChromaDB collection
            model_name: Name of the embedding model to use
        """
        # Initialize ChromaDB client (local, persistent)
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Load embedding model (free, runs locally)
        print(f"Loading embedding model: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"Model {model_name} loaded!")
    
    def load_document(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Load a document, split it into chunks, and store in ChromaDB
        """
        print(f"Loading document from {file_path}...")
        
        # Read the document
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple chunking strategy (split by paragraphs, then by sentences if needed)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph would exceed chunk_size, save current chunk
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = ' '.join(words[-chunk_overlap:]) if len(words) > chunk_overlap else current_chunk
                current_chunk = overlap_text + " " + para
            else:
                current_chunk += " " + para if current_chunk else para
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        print(f"Split document into {len(chunks)} chunks")
        
        # Generate embeddings and store in ChromaDB
        print("Generating embeddings and storing in ChromaDB...")
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        # Store in ChromaDB
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"chunk_id": i, "source": file_path} for i in range(len(chunks))]
        )
        
        print(f"Stored {len(chunks)} chunks in ChromaDB")
        return len(chunks)
    
    def query(self, question: str, top_k: int = 3, verbose: bool = True) -> Tuple[str, List[str]]:
        """
        Query the RAG system with a question
        Returns: (answer, list of retrieved contexts)
        
        Args:
            question: The question to ask
            top_k: Number of top results to retrieve
            verbose: Whether to print retrieval details
        """
        if verbose:
            print(f"\nQuery: {question}")
        
        # Generate embedding for the question
        query_embedding = self.embedding_model.encode([question]).tolist()[0]
        
        # Retrieve similar chunks
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Extract retrieved documents
        retrieved_docs = results['documents'][0] if results['documents'] else []
        
        if verbose:
            print(f"\nRetrieved {len(retrieved_docs)} relevant chunks:")
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"\n--- Chunk {i} ---")
                print(doc[:200] + "..." if len(doc) > 200 else doc)
        
        # Simple answer generation: combine retrieved contexts
        # In a production system, you'd use an LLM here (like OpenAI, Anthropic, or local model)
        context = "\n\n".join(retrieved_docs)
        
        # For this simple version, we'll return the most relevant chunk as the answer
        # You can enhance this by adding an LLM call here
        answer = f"Based on the retrieved information:\n\n{retrieved_docs[0] if retrieved_docs else 'No relevant information found.'}"
        
        return answer, retrieved_docs
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection cleared!")


def run_benchmark():
    """
    Benchmark function to compare three embedding models
    """
    # Models to test
    models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "BAAI/bge-small-en-v1.5"
    ]
    
    # Questions to test
    questions = [
        "What is the main difference between Naive and Advanced RAG?",
        "How does HyDE improve retrieval?",
        "Explain the concept of a Semantic Router."
    ]
    
    doc_path = "sample_document.txt"
    if not os.path.exists(doc_path):
        print(f"Error: {doc_path} not found!")
        return
    
    print("=" * 80)
    print("RAG EMBEDDING MODEL BENCHMARK")
    print("=" * 80)
    
    # Loop through each model
    for model_name in models:
        print("\n" + "=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)
        
        # Create a unique collection name for this model
        collection_name = f"benchmark_{model_name.replace('/', '_').replace('-', '_')}"
        
        # Initialize RAG with this model
        rag = SimpleRAG(collection_name=collection_name, model_name=model_name)
        
        # Load document into this model's collection
        rag.load_document(doc_path)
        
        # Test each question
        for i, question in enumerate(questions, 1):
            print("\n" + "-" * 80)
            print(f"QUESTION {i}: {question}")
            print("-" * 80)
            
            # Query and get results
            answer, retrieved_chunks = rag.query(question, top_k=3, verbose=False)
            
            # Print retrieved chunks for comparison
            print(f"\nRetrieved Chunks (Top 3):")
            for j, chunk in enumerate(retrieved_chunks, 1):
                print(f"\n--- Chunk {j} ---")
                print(chunk)
                print()
        
        # Clean up: delete the temporary collection
        try:
            rag.client.delete_collection(name=collection_name)
            print(f"\nCleaned up collection: {collection_name}")
        except:
            pass
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print("\nCompare the retrieved chunks above to see how each model performs!")


def main():
    """
    Main function to demonstrate the RAG system
    """
    print("=" * 60)
    print("Simple RAG System")
    print("=" * 60)
    
    # Initialize RAG system
    rag = SimpleRAG()
    
    # Load the sample document
    doc_path = "sample_document.txt"
    if os.path.exists(doc_path):
        rag.load_document(doc_path)
    else:
        print(f"Error: {doc_path} not found!")
        return
    
    # Example queries
    print("\n" + "=" * 60)
    print("Example Queries")
    print("=" * 60)
    
    questions = [
        "What is RAG?",
        "How does RAG work?",
        "What are the steps in a RAG pipeline?",
        "What vector databases are commonly used?",
        "What factors affect RAG quality?",
    ]
    
    for question in questions:
        answer, contexts = rag.query(question, top_k=2)
        print("\n" + "-" * 60)
        print(f"Q: {question}")
        print(f"\nA: {answer}")
        print("-" * 60)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode - Ask questions about RAG!")
    print("Type 'quit' or 'exit' to stop")
    print("=" * 60)
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if question:
            answer, contexts = rag.query(question, top_k=3)
            print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    import sys
    
    # Check if user wants to run benchmark
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        run_benchmark()
    else:
        main()

