"""
Cache Benchmark Script
Tests cached vs uncached RAG performance with realistic query patterns.
"""

from __future__ import annotations

import time
import json
from datetime import datetime
from typing import List, Dict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from cached_rag import CachedRAG
from rag_system import SimpleRAG


def run_benchmark():
    """Run comprehensive cache benchmark"""
    console = Console()
    
    console.print()
    console.print(Panel(
        "[bold]Cache Benchmark Suite[/bold]\n"
        "Comparing cached vs uncached RAG performance",
        box=box.DOUBLE,
        border_style="cyan"
    ))
    
    # Test queries: 10 unique, 10 repeats (simulating 30-40% repeat rate)
    unique_queries = [
        "What is RAG?",
        "How does RAG work?",
        "What are the benefits of RAG?",
        "Explain chunking in RAG",
        "What is an embedding model?",
        "How does vector search work?",
        "What is ChromaDB?",
        "Explain semantic similarity",
        "What are RAG use cases?",
        "How to improve RAG accuracy?",
    ]
    
    # Repeat some queries to simulate real-world patterns
    repeat_queries = [
        "What is RAG?",  # Repeat 1
        "What is RAG?",  # Repeat 2
        "How does RAG work?",  # Repeat 1
        "What are the benefits of RAG?",  # Repeat 1
        "What is RAG?",  # Repeat 3
        "Explain chunking in RAG",  # Repeat 1
        "What is RAG?",  # Repeat 4
        "How does RAG work?",  # Repeat 2
        "What is RAG?",  # Repeat 5
        "What are the benefits of RAG?",  # Repeat 2
    ]
    
    all_queries = unique_queries + repeat_queries
    
    console.print()
    console.print(Panel(
        "[bold]Benchmark Configuration[/bold]\n"
        f"Total queries: {len(all_queries)}\n"
        f"Unique queries: {len(unique_queries)}\n"
        f"Repeat queries: {len(repeat_queries)}\n"
        f"Repeat rate: {len(repeat_queries) / len(all_queries) * 100:.1f}%",
        box=box.ROUNDED,
        border_style="yellow"
    ))
    
    # Test 1: Uncached RAG (baseline)
    console.print()
    console.print(Panel(
        "[bold]Test 1: Uncached RAG (Baseline)[/bold]",
        box=box.ROUNDED,
        border_style="red"
    ))
    
    uncached_rag = SimpleRAG(collection_name="rag_documents")
    uncached_times = []
    uncached_start = time.time()
    
    for i, query in enumerate(all_queries, 1):
        query_start = time.time()
        answer, chunks = uncached_rag.query(query, verbose=False)
        query_time = (time.time() - query_start) * 1000
        uncached_times.append(query_time)
        console.print(f"[dim]Query {i}/{len(all_queries)}: {query_time:.2f}ms[/dim]")
    
    uncached_total = (time.time() - uncached_start) * 1000
    uncached_avg = sum(uncached_times) / len(uncached_times)
    
    # Test 2: Cached RAG
    console.print()
    console.print(Panel(
        "[bold]Test 2: Cached RAG (3-Tier Cache)[/bold]",
        box=box.ROUNDED,
        border_style="green"
    ))
    
    cached_rag = CachedRAG(
        collection_name="rag_documents",
        exact_cache_size=100,
        semantic_cache_size=200,
        semantic_threshold=0.95
    )
    
    cached_times = []
    cached_start = time.time()
    
    for i, query in enumerate(all_queries, 1):
        query_start = time.time()
        answer, chunks, metadata = cached_rag.query(query, verbose=False)
        query_time = (time.time() - query_start) * 1000
        cached_times.append(query_time)
        
        tier_icon = {
            "exact": "⚡",
            "semantic": "🔍",
            "full_rag": "🔄"
        }.get(metadata["tier_used"], "❓")
        
        console.print(f"[dim]{tier_icon} Query {i}/{len(all_queries)}: {query_time:.2f}ms ({metadata['tier_used']})[/dim]")
    
    cached_total = (time.time() - cached_start) * 1000
    cached_avg = sum(cached_times) / len(cached_times)
    
    # Get cache stats
    cache_stats = cached_rag.get_stats()
    
    # Display comparison
    console.print()
    console.print(Panel(
        "[bold]BENCHMARK RESULTS[/bold]",
        box=box.DOUBLE,
        border_style="cyan"
    ))
    console.print()
    
    # Comparison table
    comparison_table = Table(
        title="Performance Comparison",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED
    )
    
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("Uncached", justify="right", style="red")
    comparison_table.add_column("Cached", justify="right", style="green")
    comparison_table.add_column("Improvement", justify="right", style="yellow")
    
    total_improvement = ((uncached_total - cached_total) / uncached_total * 100)
    avg_improvement = ((uncached_avg - cached_avg) / uncached_avg * 100)
    
    comparison_table.add_row(
        "Total Time",
        f"{uncached_total:.2f} ms",
        f"{cached_total:.2f} ms",
        f"{total_improvement:.1f}% faster"
    )
    
    comparison_table.add_row(
        "Average Latency",
        f"{uncached_avg:.2f} ms",
        f"{cached_avg:.2f} ms",
        f"{avg_improvement:.1f}% faster"
    )
    
    comparison_table.add_row(
        "Cache Hit Rate",
        "N/A",
        f"{cache_stats['hit_rate']:.1f}%",
        f"{cache_stats['exact_hit_rate']:.1f}% exact, {cache_stats['semantic_hit_rate']:.1f}% semantic"
    )
    
    comparison_table.add_row(
        "Full RAG Calls",
        f"{len(all_queries)}",
        f"{cache_stats['full_rag_calls']}",
        f"{(1 - cache_stats['full_rag_calls'] / len(all_queries)) * 100:.1f}% reduction"
    )
    
    console.print(comparison_table)
    console.print()
    
    # Cache breakdown
    cache_breakdown_table = Table(
        title="Cache Performance Breakdown",
        show_header=True,
        header_style="bold blue",
        box=box.ROUNDED
    )
    
    cache_breakdown_table.add_column("Tier", style="cyan")
    cache_breakdown_table.add_column("Hits", justify="right", style="green")
    cache_breakdown_table.add_column("Hit Rate", justify="right", style="yellow")
    cache_breakdown_table.add_column("Avg Latency", justify="right", style="magenta")
    
    cache_breakdown_table.add_row(
        "Exact Match",
        str(cache_stats["exact_hits"]),
        f"{cache_stats['exact_hit_rate']:.1f}%",
        f"{cache_stats['avg_cached_latency_ms']:.2f} ms"
    )
    
    cache_breakdown_table.add_row(
        "Semantic Similarity",
        str(cache_stats["semantic_hits"]),
        f"{cache_stats['semantic_hit_rate']:.1f}%",
        f"{cache_stats['avg_cached_latency_ms']:.2f} ms"
    )
    
    cache_breakdown_table.add_row(
        "Full RAG",
        str(cache_stats["full_rag_calls"]),
        f"{(cache_stats['full_rag_calls'] / cache_stats['total_queries'] * 100):.1f}%",
        f"{cache_stats['avg_uncached_latency_ms']:.2f} ms"
    )
    
    console.print(cache_breakdown_table)
    console.print()
    
    # Cost analysis (estimated)
    # Assuming $0.0001 per full RAG call
    cost_per_call = 0.0001
    uncached_cost = len(all_queries) * cost_per_call
    cached_cost = cache_stats["full_rag_calls"] * cost_per_call
    cost_savings = uncached_cost - cached_cost
    cost_reduction = (cost_savings / uncached_cost * 100) if uncached_cost > 0 else 0
    
    cost_table = Table(
        title="Cost Analysis (Estimated)",
        show_header=True,
        header_style="bold green",
        box=box.ROUNDED
    )
    
    cost_table.add_column("Metric", style="cyan")
    cost_table.add_column("Value", justify="right", style="yellow")
    
    cost_table.add_row("Uncached Cost", f"${uncached_cost:.4f}")
    cost_table.add_row("Cached Cost", f"${cached_cost:.4f}")
    cost_table.add_row("Cost Savings", f"${cost_savings:.4f}")
    cost_table.add_row("Cost Reduction", f"{cost_reduction:.1f}%")
    
    console.print(cost_table)
    console.print()
    
    # Save detailed analysis
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "benchmark_config": {
            "total_queries": len(all_queries),
            "unique_queries": len(unique_queries),
            "repeat_queries": len(repeat_queries),
            "repeat_rate": len(repeat_queries) / len(all_queries) * 100
        },
        "uncached_performance": {
            "total_time_ms": uncached_total,
            "average_latency_ms": uncached_avg,
            "total_queries": len(all_queries),
            "estimated_cost": uncached_cost
        },
        "cached_performance": {
            "total_time_ms": cached_total,
            "average_latency_ms": cached_avg,
            "cache_stats": cache_stats,
            "estimated_cost": cached_cost
        },
        "improvements": {
            "total_time_reduction": total_improvement,
            "average_latency_reduction": avg_improvement,
            "cost_reduction": cost_reduction,
            "cache_hit_rate": cache_stats["hit_rate"]
        },
        "query_history": cached_rag.stats["query_history"]
    }
    
    with open("cache_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Create markdown report
    with open("cache_analysis.md", "w") as f:
        f.write("# Cache Performance Analysis\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total Queries:** {len(all_queries)}\n")
        f.write(f"- **Repeat Rate:** {len(repeat_queries) / len(all_queries) * 100:.1f}%\n")
        f.write(f"- **Cache Hit Rate:** {cache_stats['hit_rate']:.1f}%\n")
        f.write(f"- **Total Time Reduction:** {total_improvement:.1f}%\n")
        f.write(f"- **Average Latency Reduction:** {avg_improvement:.1f}%\n")
        f.write(f"- **Cost Reduction:** {cost_reduction:.1f}%\n\n")
        f.write("## Performance Comparison\n\n")
        f.write("| Metric | Uncached | Cached | Improvement |\n")
        f.write("|--------|----------|--------|-------------|\n")
        f.write(f"| Total Time | {uncached_total:.2f} ms | {cached_total:.2f} ms | {total_improvement:.1f}% faster |\n")
        f.write(f"| Average Latency | {uncached_avg:.2f} ms | {cached_avg:.2f} ms | {avg_improvement:.1f}% faster |\n")
        f.write(f"| Full RAG Calls | {len(all_queries)} | {cache_stats['full_rag_calls']} | {(1 - cache_stats['full_rag_calls'] / len(all_queries)) * 100:.1f}% reduction |\n\n")
        f.write("## Cache Breakdown\n\n")
        f.write("| Tier | Hits | Hit Rate | Avg Latency |\n")
        f.write("|------|------|----------|-------------|\n")
        f.write(f"| Exact Match | {cache_stats['exact_hits']} | {cache_stats['exact_hit_rate']:.1f}% | {cache_stats['avg_cached_latency_ms']:.2f} ms |\n")
        f.write(f"| Semantic Similarity | {cache_stats['semantic_hits']} | {cache_stats['semantic_hit_rate']:.1f}% | {cache_stats['avg_cached_latency_ms']:.2f} ms |\n")
        f.write(f"| Full RAG | {cache_stats['full_rag_calls']} | {(cache_stats['full_rag_calls'] / cache_stats['total_queries'] * 100):.1f}% | {cache_stats['avg_uncached_latency_ms']:.2f} ms |\n\n")
        f.write("## Cost Analysis\n\n")
        f.write(f"- **Uncached Cost:** ${uncached_cost:.4f}\n")
        f.write(f"- **Cached Cost:** ${cached_cost:.4f}\n")
        f.write(f"- **Cost Savings:** ${cost_savings:.4f}\n")
        f.write(f"- **Cost Reduction:** {cost_reduction:.1f}%\n\n")
    
    console.print()
    console.print(f"[green]✓ Analysis saved to cache_analysis.json and cache_analysis.md[/green]")
    console.print()


if __name__ == "__main__":
    run_benchmark()
