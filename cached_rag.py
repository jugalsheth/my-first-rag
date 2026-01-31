"""
Cached RAG System with 3-Tier Caching
Implements exact match cache, semantic similarity cache, and full RAG pipeline.

Tier 1: Exact Match Cache (hash-based, < 1ms)
Tier 2: Semantic Similarity Cache (embedding-based, < 100ms)
Tier 3: Full RAG Pipeline (2000ms)

This reduces latency and cost for repeated or similar queries.
"""

from __future__ import annotations

import os
import sys
import time
import hashlib
import json
import warnings
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

# Early progress messages
print("💾 Cached RAG System (3-Tier Caching)")
print("Loading dependencies...")
sys.stdout.flush()

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from sentence_transformers import SentenceTransformer

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Import base RAG system
try:
    from rag_system import SimpleRAG
except ImportError:
    print("Warning: rag_system.py not found. Caching will work but needs a RAG backend.")
    SimpleRAG = None


class CachedRAG:
    """RAG system with 3-tier caching"""
    
    def __init__(
        self,
        rag_backend=None,
        collection_name: str = "rag_documents",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        exact_cache_size: int = 100,
        semantic_cache_size: int = 200,
        exact_cache_ttl: int = 3600,  # 1 hour
        semantic_cache_ttl: int = 3600,  # 1 hour
        semantic_threshold: float = 0.95,  # Similarity threshold for semantic cache
    ):
        """
        Initialize Cached RAG system
        
        Args:
            rag_backend: RAG system to wrap (SimpleRAG or compatible)
            collection_name: ChromaDB collection name
            embedding_model_name: Embedding model for semantic cache
            exact_cache_size: Max entries in exact match cache (LRU)
            semantic_cache_size: Max entries in semantic cache (LRU)
            exact_cache_ttl: Time-to-live for exact cache entries (seconds)
            semantic_cache_ttl: Time-to-live for semantic cache entries (seconds)
            semantic_threshold: Minimum similarity for semantic cache hit (0-1)
        """
        self.console = Console()
        
        # Initialize RAG backend
        if rag_backend is None:
            if SimpleRAG:
                self.rag = SimpleRAG(collection_name=collection_name, model_name=embedding_model_name)
            else:
                raise ValueError("No RAG backend available. Please provide one or ensure rag_system.py exists.")
        else:
            self.rag = rag_backend
        
        # Load embedding model for semantic cache
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Tier 1: Exact match cache (hash-based)
        # Structure: {query_hash: {answer, chunks, timestamp, hit_count}}
        self.exact_cache: OrderedDict = OrderedDict()
        self.exact_cache_size = exact_cache_size
        self.exact_cache_ttl = exact_cache_ttl
        
        # Tier 2: Semantic similarity cache (embedding-based)
        # Structure: [{query, embedding, answer, chunks, timestamp, hit_count}, ...]
        self.semantic_cache: List[Dict] = []
        self.semantic_cache_size = semantic_cache_size
        self.semantic_cache_ttl = semantic_cache_ttl
        self.semantic_threshold = semantic_threshold
        
        # Performance tracking
        self.stats = {
            "total_queries": 0,
            "exact_hits": 0,
            "semantic_hits": 0,
            "full_rag_calls": 0,
            "total_latency_ms": 0.0,
            "cached_latency_ms": 0.0,
            "uncached_latency_ms": 0.0,
            "query_history": []
        }
        
        self.console.print("[green]✓[/green] Cached RAG initialized")
        self.console.print(f"[dim]Exact cache: {exact_cache_size} entries, TTL: {exact_cache_ttl}s[/dim]")
        self.console.print(f"[dim]Semantic cache: {semantic_cache_size} entries, TTL: {semantic_cache_ttl}s[/dim]")
        self.console.print(f"[dim]Semantic threshold: {semantic_threshold}[/dim]")
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for exact match cache"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def _check_exact_cache(self, query: str) -> Optional[Dict]:
        """Check Tier 1: Exact match cache"""
        query_hash = self._hash_query(query)
        
        if query_hash in self.exact_cache:
            entry = self.exact_cache[query_hash]
            
            # Check TTL
            age = time.time() - entry["timestamp"]
            if age > self.exact_cache_ttl:
                # Expired, remove it
                del self.exact_cache[query_hash]
                return None
            
            # Cache hit! Move to end (LRU)
            self.exact_cache.move_to_end(query_hash)
            entry["hit_count"] += 1
            
            return entry
        
        return None
    
    def _check_semantic_cache(self, query: str) -> Optional[Dict]:
        """Check Tier 2: Semantic similarity cache"""
        if not self.semantic_cache:
            return None
        
        # Embed the query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Check similarity with cached queries
        best_match = None
        best_similarity = 0.0
        best_index = -1
        
        current_time = time.time()
        
        for i, entry in enumerate(self.semantic_cache):
            # Check TTL
            age = current_time - entry["timestamp"]
            if age > self.semantic_cache_ttl:
                continue  # Skip expired entries
            
            # Calculate cosine similarity
            cached_embedding = entry["embedding"]
            similarity = np.dot(query_embedding, cached_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
            )
            
            if similarity > best_similarity and similarity >= self.semantic_threshold:
                best_similarity = similarity
                best_match = entry
                best_index = i
        
        if best_match:
            # Cache hit! Update hit count and move to end (LRU)
            best_match["hit_count"] += 1
            if best_index >= 0:
                # Move to end for LRU
                self.semantic_cache.pop(best_index)
                self.semantic_cache.append(best_match)
            
            best_match["similarity"] = best_similarity
            return best_match
        
        return None
    
    def _add_to_exact_cache(self, query: str, answer: str, chunks: List[str]):
        """Add entry to exact match cache"""
        query_hash = self._hash_query(query)
        
        entry = {
            "query": query,
            "answer": answer,
            "chunks": chunks,
            "timestamp": time.time(),
            "hit_count": 0
        }
        
        # LRU eviction
        if len(self.exact_cache) >= self.exact_cache_size:
            # Remove oldest entry
            self.exact_cache.popitem(last=False)
        
        self.exact_cache[query_hash] = entry
    
    def _add_to_semantic_cache(self, query: str, answer: str, chunks: List[str]):
        """Add entry to semantic similarity cache"""
        # Embed the query
        query_embedding = self.embedding_model.encode([query])[0]
        
        entry = {
            "query": query,
            "embedding": query_embedding,
            "answer": answer,
            "chunks": chunks,
            "timestamp": time.time(),
            "hit_count": 0
        }
        
        # LRU eviction
        if len(self.semantic_cache) >= self.semantic_cache_size:
            # Remove oldest entry
            self.semantic_cache.pop(0)
        
        self.semantic_cache.append(entry)
    
    def query(
        self,
        question: str,
        top_k: int = 3,
        verbose: bool = True
    ) -> Tuple[str, List[str], Dict]:
        """
        Query with 3-tier caching
        
        Returns:
            (answer, chunks, metadata) tuple
            metadata includes: tier_used, latency_ms, cache_hit, similarity (if semantic)
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        metadata = {
            "tier_used": None,
            "latency_ms": 0.0,
            "cache_hit": False,
            "similarity": None
        }
        
        # Tier 1: Exact match cache
        exact_entry = self._check_exact_cache(question)
        if exact_entry:
            latency_ms = (time.time() - start_time) * 1000
            metadata["tier_used"] = "exact"
            metadata["latency_ms"] = latency_ms
            metadata["cache_hit"] = True
            
            self.stats["exact_hits"] += 1
            self.stats["total_latency_ms"] += latency_ms
            self.stats["cached_latency_ms"] += latency_ms
            
            if verbose:
                self.console.print(f"[green]✓ Exact cache hit![/green] (latency: {latency_ms:.2f}ms)")
            
            self.stats["query_history"].append({
                "query": question,
                "tier": "exact",
                "latency_ms": latency_ms,
                "timestamp": datetime.now().isoformat()
            })
            
            return exact_entry["answer"], exact_entry["chunks"], metadata
        
        # Tier 2: Semantic similarity cache
        semantic_entry = self._check_semantic_cache(question)
        if semantic_entry:
            latency_ms = (time.time() - start_time) * 1000
            metadata["tier_used"] = "semantic"
            metadata["latency_ms"] = latency_ms
            metadata["cache_hit"] = True
            metadata["similarity"] = semantic_entry.get("similarity", 0.0)
            
            self.stats["semantic_hits"] += 1
            self.stats["total_latency_ms"] += latency_ms
            self.stats["cached_latency_ms"] += latency_ms
            
            if verbose:
                similarity = semantic_entry.get("similarity", 0.0)
                self.console.print(f"[yellow]✓ Semantic cache hit![/yellow] (similarity: {similarity:.3f}, latency: {latency_ms:.2f}ms)")
            
            self.stats["query_history"].append({
                "query": question,
                "tier": "semantic",
                "latency_ms": latency_ms,
                "similarity": similarity,
                "timestamp": datetime.now().isoformat()
            })
            
            return semantic_entry["answer"], semantic_entry["chunks"], metadata
        
        # Tier 3: Full RAG pipeline
        if verbose:
            self.console.print("[dim]Cache miss, running full RAG pipeline...[/dim]")
        
        rag_start = time.time()
        answer, chunks = self.rag.query(question, top_k=top_k, verbose=False)
        rag_latency_ms = (time.time() - rag_start) * 1000
        
        total_latency_ms = (time.time() - start_time) * 1000
        metadata["tier_used"] = "full_rag"
        metadata["latency_ms"] = total_latency_ms
        metadata["cache_hit"] = False
        
        self.stats["full_rag_calls"] += 1
        self.stats["total_latency_ms"] += total_latency_ms
        self.stats["uncached_latency_ms"] += total_latency_ms
        
        # Add to both caches
        self._add_to_exact_cache(question, answer, chunks)
        self._add_to_semantic_cache(question, answer, chunks)
        
        if verbose:
            self.console.print(f"[blue]Full RAG completed[/blue] (latency: {total_latency_ms:.2f}ms)")
        
        self.stats["query_history"].append({
            "query": question,
            "tier": "full_rag",
            "latency_ms": total_latency_ms,
            "timestamp": datetime.now().isoformat()
        })
        
        return answer, chunks, metadata
    
    def get_stats(self) -> Dict:
        """Get cache performance statistics"""
        total = self.stats["total_queries"]
        if total == 0:
            return {
                "total_queries": 0,
                "hit_rate": 0.0,
                "exact_hit_rate": 0.0,
                "semantic_hit_rate": 0.0,
                "avg_latency_ms": 0.0,
                "avg_cached_latency_ms": 0.0,
                "avg_uncached_latency_ms": 0.0,
                "latency_reduction": 0.0,
                "exact_cache_size": len(self.exact_cache),
                "semantic_cache_size": len(self.semantic_cache)
            }
        
        exact_hits = self.stats["exact_hits"]
        semantic_hits = self.stats["semantic_hits"]
        total_hits = exact_hits + semantic_hits
        
        avg_latency = self.stats["total_latency_ms"] / total
        avg_cached = self.stats["cached_latency_ms"] / total_hits if total_hits > 0 else 0.0
        avg_uncached = self.stats["uncached_latency_ms"] / self.stats["full_rag_calls"] if self.stats["full_rag_calls"] > 0 else 0.0
        
        latency_reduction = ((avg_uncached - avg_cached) / avg_uncached * 100) if avg_uncached > 0 else 0.0
        
        return {
            "total_queries": total,
            "exact_hits": exact_hits,
            "semantic_hits": semantic_hits,
            "full_rag_calls": self.stats["full_rag_calls"],
            "hit_rate": (total_hits / total * 100) if total > 0 else 0.0,
            "exact_hit_rate": (exact_hits / total * 100) if total > 0 else 0.0,
            "semantic_hit_rate": (semantic_hits / total * 100) if total > 0 else 0.0,
            "avg_latency_ms": avg_latency,
            "avg_cached_latency_ms": avg_cached,
            "avg_uncached_latency_ms": avg_uncached,
            "latency_reduction": latency_reduction,
            "exact_cache_size": len(self.exact_cache),
            "semantic_cache_size": len(self.semantic_cache)
        }
    
    def display_stats(self):
        """Display cache performance statistics"""
        stats = self.get_stats()
        
        self.console.print()
        self.console.print(Panel(
            "[bold]CACHE PERFORMANCE STATISTICS[/bold]",
            box=box.DOUBLE,
            border_style="cyan"
        ))
        self.console.print()
        
        # Summary table
        summary_table = Table(
            title="Cache Performance Summary",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )
        
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="green")
        
        summary_table.add_row("Total Queries", str(stats["total_queries"]))
        summary_table.add_row("Exact Cache Hits", f"{stats['exact_hits']} ({stats['exact_hit_rate']:.1f}%)")
        summary_table.add_row("Semantic Cache Hits", f"{stats['semantic_hits']} ({stats['semantic_hit_rate']:.1f}%)")
        summary_table.add_row("Full RAG Calls", str(stats["full_rag_calls"]))
        summary_table.add_row("Overall Hit Rate", f"{stats['hit_rate']:.1f}%")
        
        self.console.print(summary_table)
        self.console.print()
        
        # Latency table
        latency_table = Table(
            title="Latency Analysis",
            show_header=True,
            header_style="bold blue",
            box=box.ROUNDED
        )
        
        latency_table.add_column("Metric", style="cyan")
        latency_table.add_column("Value", justify="right", style="yellow")
        
        latency_table.add_row("Average Latency (All)", f"{stats['avg_latency_ms']:.2f} ms")
        latency_table.add_row("Average Latency (Cached)", f"{stats['avg_cached_latency_ms']:.2f} ms")
        latency_table.add_row("Average Latency (Uncached)", f"{stats['avg_uncached_latency_ms']:.2f} ms")
        latency_table.add_row("Latency Reduction", f"{stats['latency_reduction']:.1f}%")
        
        self.console.print(latency_table)
        self.console.print()
        
        # Cache size table
        cache_table = Table(
            title="Cache Status",
            show_header=True,
            header_style="bold green",
            box=box.ROUNDED
        )
        
        cache_table.add_column("Cache Tier", style="cyan")
        cache_table.add_column("Current Size", justify="right", style="yellow")
        cache_table.add_column("Max Size", justify="right", style="dim")
        
        cache_table.add_row("Exact Match Cache", str(stats["exact_cache_size"]), str(self.exact_cache_size))
        cache_table.add_row("Semantic Cache", str(stats["semantic_cache_size"]), str(self.semantic_cache_size))
        
        self.console.print(cache_table)
        self.console.print()
    
    def clear_cache(self):
        """Clear all caches"""
        self.exact_cache.clear()
        self.semantic_cache.clear()
        self.console.print("[yellow]Cache cleared[/yellow]")


def main():
    """Main function for testing"""
    console = Console()
    
    console.print()
    console.print(Panel(
        "[bold]Cached RAG System - Test Suite[/bold]\n"
        "Testing 3-tier caching with repeated and similar queries",
        box=box.ROUNDED,
        border_style="cyan"
    ))
    
    # Initialize cached RAG
    cached_rag = CachedRAG(
        collection_name="rag_documents",
        exact_cache_size=100,
        semantic_cache_size=200,
        semantic_threshold=0.95
    )
    
    # Test queries
    test_queries = [
        "What is RAG?",
        "What is RAG?",  # Exact duplicate
        "What is RAG?",  # Exact duplicate again
        "Explain RAG",  # Semantic similar
        "What does RAG mean?",  # Semantic similar
        "How does RAG work?",  # Different
        "What is RAG?",  # Exact duplicate again
        "Tell me about RAG",  # Semantic similar
    ]
    
    console.print()
    console.print(Panel(
        "[bold]Running Test Queries[/bold]\n"
        f"Total queries: {len(test_queries)}\n"
        "Expected: Mix of exact hits, semantic hits, and full RAG calls",
        box=box.ROUNDED,
        border_style="yellow"
    ))
    
    # Run queries
    for i, query in enumerate(test_queries, 1):
        console.print()
        console.print(f"[bold]Query {i}/{len(test_queries)}:[/bold] {query}")
        answer, chunks, metadata = cached_rag.query(query, verbose=True)
        
        tier_color = {
            "exact": "green",
            "semantic": "yellow",
            "full_rag": "blue"
        }.get(metadata["tier_used"], "white")
        
        console.print(f"[{tier_color}]Tier: {metadata['tier_used']}[/{tier_color}]")
        if metadata.get("similarity"):
            console.print(f"[dim]Similarity: {metadata['similarity']:.3f}[/dim]")
        console.print(f"[dim]Latency: {metadata['latency_ms']:.2f} ms[/dim]")
        console.print()
    
    # Display statistics
    cached_rag.display_stats()
    
    # Save analysis
    stats = cached_rag.get_stats()
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "test_queries": len(test_queries),
        "statistics": stats,
        "query_history": cached_rag.stats["query_history"]
    }
    
    with open("cache_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    console.print()
    console.print(f"[green]✓ Analysis saved to cache_analysis.json[/green]")


if __name__ == "__main__":
    main()
