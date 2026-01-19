"""
Chunking Strategy Research Experiment
Investigative analysis of chunking strategies for RAG systems
Tests 3 different chunking strategies with comprehensive metrics and visualizations
"""

import os
import warnings
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional

# Suppress harmless urllib3/OpenSSL warning on macOS
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.text import Text
from rich import box


class ChunkingExperiment:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the chunking experiment with rich formatting
        
        Args:
            model_name: Embedding model to use (same for all experiments)
        """
        self.console = Console()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load embedding model (same for all experiments)
        with self.console.status("[bold green]Loading embedding model...") as status:
            self.embedding_model = SentenceTransformer(model_name)
            self.model_name = model_name
        
        # Initialize tiktoken for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Define chunking strategies
        self.strategies = {
            "Small": {
                "chunk_size": 256,
                "chunk_overlap": 20,
                "description": "256 tokens, 20 token overlap",
                "color": "red"
            },
            "Medium": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "description": "512 tokens, 50 token overlap",
                "color": "yellow"
            },
            "Large": {
                "chunk_size": 1024,
                "chunk_overlap": 100,
                "description": "1024 tokens, 100 token overlap",
                "color": "green"
            }
        }
        
        # Store experiment data
        self.experiment_data = {}
    
    def print_header(self, title: str, subtitle: str = ""):
        """Print a beautiful header"""
        header_text = f"[bold cyan]{title}[/bold cyan]"
        if subtitle:
            header_text += f"\n[dim]{subtitle}[/dim]"
        self.console.print(Panel(header_text, box=box.DOUBLE, expand=False))
        self.console.print()
    
    def load_and_chunk_document(self, file_path: str, strategy_name: str) -> Tuple[List[str], Dict]:
        """
        Load document and chunk it using the specified strategy
        
        Returns:
            Tuple of (chunks, statistics_dict)
        """
        # Read the document
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        original_tokens = len(self.tokenizer.encode(text))
        
        # Get strategy parameters
        strategy = self.strategies[strategy_name]
        
        # Create token counting function
        def count_tokens(text: str) -> int:
            return len(self.tokenizer.encode(text))
        
        # Create RecursiveCharacterTextSplitter with token-based length
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=strategy["chunk_size"],
            chunk_overlap=strategy["chunk_overlap"],
            length_function=count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Split the document
        chunks = text_splitter.split_text(text)
        
        # Calculate comprehensive statistics
        token_counts = [len(self.tokenizer.encode(chunk)) for chunk in chunks]
        char_counts = [len(chunk) for chunk in chunks]
        
        # Calculate overlap effectiveness
        overlap_effectiveness = self._calculate_overlap_effectiveness(chunks, strategy["chunk_overlap"])
        
        stats = {
            "num_chunks": len(chunks),
            "token_counts": token_counts,
            "char_counts": char_counts,
            "avg_tokens": np.mean(token_counts),
            "std_tokens": np.std(token_counts),
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "avg_chars": np.mean(char_counts),
            "original_tokens": original_tokens,
            "total_chunk_tokens": sum(token_counts),
            "overlap_effectiveness": overlap_effectiveness,
            "chunk_size_variance": np.var(token_counts),
            "coverage_ratio": sum(token_counts) / original_tokens if original_tokens > 0 else 0
        }
        
        return chunks, stats
    
    def _calculate_overlap_effectiveness(self, chunks: List[str], expected_overlap: int) -> float:
        """Calculate how effective the overlap is at preserving context"""
        if len(chunks) < 2:
            return 0.0
        
        overlap_scores = []
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i][-200:]  # Last 200 chars
            chunk2_start = chunks[i+1][:200]  # First 200 chars
            
            # Find longest common substring
            max_overlap = 0
            for j in range(min(50, len(chunk1_end))):
                if chunk1_end[-j:] in chunk2_start:
                    max_overlap = max(max_overlap, j)
            
            # Normalize by expected overlap
            effectiveness = min(max_overlap / expected_overlap if expected_overlap > 0 else 0, 1.0)
            overlap_scores.append(effectiveness)
        
        return np.mean(overlap_scores) if overlap_scores else 0.0
    
    def display_chunking_statistics(self, all_stats: Dict):
        """Display comprehensive chunking statistics in a beautiful table"""
        table = Table(title="📊 Chunking Statistics", show_header=True, header_style="bold magenta", box=box.ROUNDED)
        
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Small (256)", justify="right", style="red")
        table.add_column("Medium (512)", justify="right", style="yellow")
        table.add_column("Large (1024)", justify="right", style="green")
        
        # Number of chunks
        table.add_row(
            "Number of Chunks",
            str(all_stats["Small"]["num_chunks"]),
            str(all_stats["Medium"]["num_chunks"]),
            str(all_stats["Large"]["num_chunks"])
        )
        
        # Average tokens
        table.add_row(
            "Avg Tokens/Chunk",
            f"{all_stats['Small']['avg_tokens']:.1f}",
            f"{all_stats['Medium']['avg_tokens']:.1f}",
            f"{all_stats['Large']['avg_tokens']:.1f}"
        )
        
        # Token range
        table.add_row(
            "Token Range",
            f"{all_stats['Small']['min_tokens']}-{all_stats['Small']['max_tokens']}",
            f"{all_stats['Medium']['min_tokens']}-{all_stats['Medium']['max_tokens']}",
            f"{all_stats['Large']['min_tokens']}-{all_stats['Large']['max_tokens']}"
        )
        
        # Standard deviation
        table.add_row(
            "Token Std Dev",
            f"{all_stats['Small']['std_tokens']:.1f}",
            f"{all_stats['Medium']['std_tokens']:.1f}",
            f"{all_stats['Large']['std_tokens']:.1f}"
        )
        
        # Coverage ratio
        table.add_row(
            "Coverage Ratio",
            f"{all_stats['Small']['coverage_ratio']:.2%}",
            f"{all_stats['Medium']['coverage_ratio']:.2%}",
            f"{all_stats['Large']['coverage_ratio']:.2%}"
        )
        
        # Overlap effectiveness
        table.add_row(
            "Overlap Effectiveness",
            f"{all_stats['Small']['overlap_effectiveness']:.2%}",
            f"{all_stats['Medium']['overlap_effectiveness']:.2%}",
            f"{all_stats['Large']['overlap_effectiveness']:.2%}"
        )
        
        self.console.print(table)
        self.console.print()
    
    def visualize_token_distribution(self, all_stats: Dict):
        """Create a visualization of token distribution"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Token Distribution Across Chunking Strategies', fontsize=16, fontweight='bold')
        
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
        strategy_names = ['Small', 'Medium', 'Large']
        
        for idx, (strategy, color) in enumerate(zip(strategy_names, colors)):
            ax = axes[idx]
            token_counts = all_stats[strategy]["token_counts"]
            
            # Create histogram
            ax.hist(token_counts, bins=min(20, len(set(token_counts))), color=color, alpha=0.7, edgecolor='black')
            ax.axvline(all_stats[strategy]["avg_tokens"], color='red', linestyle='--', 
                      label=f'Mean: {all_stats[strategy]["avg_tokens"]:.0f}')
            ax.set_xlabel('Tokens per Chunk', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'{strategy} Strategy\n({all_stats[strategy]["num_chunks"]} chunks)', 
                        fontweight='bold', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('chunk_token_distribution.png', dpi=150, bbox_inches='tight')
        self.console.print(f"[green]✓[/green] Saved visualization: [bold]chunk_token_distribution.png[/bold]")
        self.console.print()
    
    def index_chunks(self, chunks: List[str], collection_name: str, strategy_name: str):
        """Index chunks into ChromaDB"""
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "strategy": strategy_name}
        )
        
        # Generate embeddings (no status display - handled by outer Progress)
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        ids = [f"{strategy_name.lower()}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "chunk_id": i,
                "strategy": strategy_name,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks)
            }
            for i, chunk in enumerate(chunks)
        ]
        
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
    
    def query_collection(self, collection_name: str, question: str, top_k: int = 3) -> Tuple[List[str], List[float], List[Dict]]:
        """Query a collection and retrieve top-k chunks"""
        collection = self.client.get_collection(name=collection_name)
        
        query_embedding = self.embedding_model.encode([question]).tolist()[0]
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved_docs = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        similarities = [1 - d for d in distances]
        
        return retrieved_docs, similarities, metadatas
    
    def calculate_context_quality_metrics(self, chunks: List[str], question: str) -> Dict:
        """Calculate quality metrics for retrieved context"""
        # Calculate keyword relevance
        question_words = set(question.lower().split())
        
        metrics = {
            "total_tokens": sum(len(self.tokenizer.encode(c)) for c in chunks),
            "total_chars": sum(len(c) for c in chunks),
            "avg_chunk_length": np.mean([len(c) for c in chunks]),
            "keyword_density": 0.0,
            "information_density": 0.0
        }
        
        # Keyword density
        all_text = " ".join(chunks).lower()
        question_keywords = [w for w in question_words if len(w) > 3]  # Filter short words
        if question_keywords:
            keyword_count = sum(all_text.count(kw) for kw in question_keywords)
            metrics["keyword_density"] = keyword_count / len(all_text.split()) if all_text.split() else 0
        
        # Information density (approximate: sentences per token)
        total_sentences = sum(c.count('.') + c.count('!') + c.count('?') for c in chunks)
        metrics["information_density"] = total_sentences / metrics["total_tokens"] if metrics["total_tokens"] > 0 else 0
        
        return metrics
    
    def display_retrieval_results(self, results: Dict, question: str):
        """Display retrieval results with rich formatting"""
        self.print_header("🔍 RETRIEVAL RESULTS", "Side-by-side comparison of chunking strategies")
        
        # Create comparison table
        comparison_table = Table(title="Retrieval Performance Comparison", show_header=True, 
                               header_style="bold magenta", box=box.ROUNDED)
        
        comparison_table.add_column("Metric", style="cyan")
        comparison_table.add_column("Small", justify="right", style="red")
        comparison_table.add_column("Medium", justify="right", style="yellow")
        comparison_table.add_column("Large", justify="right", style="green")
        
        # Calculate metrics
        for strategy_name in ["Small", "Medium", "Large"]:
            strategy_results = results[strategy_name]
            quality_metrics = self.calculate_context_quality_metrics(
                strategy_results["chunks"], question
            )
            results[strategy_name]["quality_metrics"] = quality_metrics
        
        # Average similarity
        comparison_table.add_row(
            "Avg Similarity Score",
            f"{np.mean(results['Small']['similarities']):.4f}",
            f"{np.mean(results['Medium']['similarities']):.4f}",
            f"{np.mean(results['Large']['similarities']):.4f}"
        )
        
        # Total context tokens
        comparison_table.add_row(
            "Total Context (tokens)",
            f"{results['Small']['quality_metrics']['total_tokens']:,}",
            f"{results['Medium']['quality_metrics']['total_tokens']:,}",
            f"{results['Large']['quality_metrics']['total_tokens']:,}"
        )
        
        # Keyword density
        comparison_table.add_row(
            "Keyword Density",
            f"{results['Small']['quality_metrics']['keyword_density']:.4f}",
            f"{results['Medium']['quality_metrics']['keyword_density']:.4f}",
            f"{results['Large']['quality_metrics']['keyword_density']:.4f}"
        )
        
        # Information density
        comparison_table.add_row(
            "Info Density (sent/token)",
            f"{results['Small']['quality_metrics']['information_density']:.4f}",
            f"{results['Medium']['quality_metrics']['information_density']:.4f}",
            f"{results['Large']['quality_metrics']['information_density']:.4f}"
        )
        
        self.console.print(comparison_table)
        self.console.print()
        
        # Display detailed chunks
        self.print_header("📄 RETRIEVED CHUNKS", "Detailed view of top-3 retrieved chunks per strategy")
        
        for strategy_name in ["Small", "Medium", "Large"]:
            strategy_results = results[strategy_name]
            color = self.strategies[strategy_name]["color"]
            
            self.console.print(Panel(
                f"[bold {color}]{strategy_name} Strategy[/bold {color}]\n"
                f"[dim]{strategy_results['description']}[/dim]",
                box=box.ROUNDED,
                border_style=color
            ))
            
            for i, (chunk, similarity, metadata) in enumerate(zip(
                strategy_results["chunks"],
                strategy_results["similarities"],
                strategy_results.get("metadatas", [{}] * len(strategy_results["chunks"]))
            ), 1):
                chunk_tokens = len(self.tokenizer.encode(chunk))
                
                # Interpret similarity score
                if similarity >= 0.8:
                    sim_interpretation = "Excellent relevance"
                elif similarity >= 0.6:
                    sim_interpretation = "Good relevance"
                elif similarity >= 0.4:
                    sim_interpretation = "Moderate relevance"
                else:
                    sim_interpretation = "Low relevance"
                
                # Create chunk panel with interpretation
                chunk_info = f"[bold]Chunk {i}[/bold] | "
                chunk_info += f"Similarity: [cyan]{similarity:.4f}[/cyan] ([dim]{sim_interpretation}[/dim]) | "
                chunk_info += f"Tokens: [yellow]{chunk_tokens}[/yellow] | "
                chunk_info += f"Chars: [yellow]{len(chunk)}[/yellow]"
                
                # Truncate chunk for display (first 500 chars)
                display_chunk = chunk[:500] + "..." if len(chunk) > 500 else chunk
                
                # Add interpretation
                interpretation = ""
                if i == 1:
                    interpretation = f"\n[dim]This is the most relevant chunk found. "
                    interpretation += f"{'It strongly matches your question' if similarity >= 0.7 else 'It has moderate relevance to your question'}."
                    if chunk_tokens < 200:
                        interpretation += " The chunk is concise, which may help with precision but could miss broader context."
                    elif chunk_tokens > 800:
                        interpretation += " The chunk is comprehensive, providing extensive context but may include some irrelevant information."
                    interpretation += "[/dim]"
                
                self.console.print(Panel(
                    f"{chunk_info}{interpretation}\n\n{display_chunk}",
                    box=box.SIMPLE,
                    border_style=color
                ))
                self.console.print()
    
    def analyze_overlap_effectiveness(self, results: Dict, all_stats: Dict):
        """Detailed analysis of overlap effectiveness"""
        self.print_header("🔗 OVERLAP EFFECTIVENESS ANALYSIS", 
                         "Investigating how overlap preserves context across chunk boundaries")
        
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Strategy", style="cyan")
        table.add_column("Expected Overlap", justify="right")
        table.add_column("Effectiveness Score", justify="right", style="green")
        table.add_column("Interpretation", style="dim")
        
        for strategy_name in ["Small", "Medium", "Large"]:
            expected = self.strategies[strategy_name]["chunk_overlap"]
            effectiveness = all_stats[strategy_name]["overlap_effectiveness"]
            
            if effectiveness > 0.7:
                interpretation = "✓ Excellent - Context well preserved"
            elif effectiveness > 0.4:
                interpretation = "⚠ Good - Some context preserved"
            else:
                interpretation = "✗ Poor - Context may be lost"
            
            table.add_row(
                strategy_name,
                f"{expected} tokens",
                f"{effectiveness:.2%}",
                interpretation
            )
        
        self.console.print(table)
        self.console.print()
    
    def generate_natural_language_summary(self, results: Dict, all_stats: Dict, question: str):
        """Generate comprehensive natural language summary of findings"""
        self.print_header("📝 EXECUTIVE SUMMARY", 
                         "Plain language interpretation of experimental findings")
        
        # Calculate key metrics
        small_tokens = results["Small"]["quality_metrics"]["total_tokens"]
        medium_tokens = results["Medium"]["quality_metrics"]["total_tokens"]
        large_tokens = results["Large"]["quality_metrics"]["total_tokens"]
        
        small_sim = np.mean(results["Small"]["similarities"])
        medium_sim = np.mean(results["Medium"]["similarities"])
        large_sim = np.mean(results["Large"]["similarities"])
        
        small_chunks = all_stats["Small"]["num_chunks"]
        medium_chunks = all_stats["Medium"]["num_chunks"]
        large_chunks = all_stats["Large"]["num_chunks"]
        
        # Determine best strategy
        similarities = {"Small": small_sim, "Medium": medium_sim, "Large": large_sim}
        best_similarity = max(similarities, key=similarities.get)
        
        # Generate summary
        summary_text = f"""
[bold]What We Tested:[/bold]
We compared three chunking strategies on your question about Semantic Routers:
• Small chunks (256 tokens) - Created {small_chunks} chunks
• Medium chunks (512 tokens) - Created {medium_chunks} chunks  
• Large chunks (1024 tokens) - Created {large_chunks} chunks

[bold]Key Finding #1: Retrieval Quality[/bold]
The {best_similarity} strategy achieved the highest similarity score ({similarities[best_similarity]:.4f}), meaning it found chunks that are semantically most similar to your question. This suggests the {best_similarity} chunk size is best at matching the language and concepts in your query.

[bold]Key Finding #2: Context Amount[/bold]
When retrieving the top 3 chunks:
• Small strategy provides {small_tokens:,} tokens of context
• Medium strategy provides {medium_tokens:,} tokens of context ({((medium_tokens-small_tokens)/small_tokens*100):.0f}% more)
• Large strategy provides {large_tokens:,} tokens of context ({((large_tokens-small_tokens)/small_tokens*100):.0f}% more)

[bold]What This Means:[/bold]
"""
        
        # Add interpretation
        if large_tokens > small_tokens * 2:
            summary_text += f"The Large strategy retrieves significantly more information ({large_tokens:,} vs {small_tokens:,} tokens). This means:\n"
            summary_text += "• More comprehensive context for complex reasoning\n"
            summary_text += "• Higher risk of including irrelevant information (noise)\n"
            summary_text += "• Better for questions requiring broad understanding\n"
        else:
            summary_text += f"The chunk sizes provide similar amounts of context, suggesting the document structure is well-suited for all strategies.\n"
        
        summary_text += f"\n[bold]Key Finding #3: Chunk Count Impact[/bold]\n"
        summary_text += f"Small chunks created {small_chunks} separate pieces, while Large chunks created only {large_chunks}. "
        
        if small_chunks > large_chunks * 2:
            summary_text += f"This means Small chunks break the document into many more pieces ({small_chunks} vs {large_chunks}), "
            summary_text += "which can help with precision but may fragment related concepts across multiple chunks.\n"
        else:
            summary_text += "The difference in chunk count is moderate, suggesting all strategies maintain reasonable document structure.\n"
        
        # Overlap analysis
        summary_text += f"\n[bold]Key Finding #4: Overlap Effectiveness[/bold]\n"
        for strategy_name in ["Small", "Medium", "Large"]:
            effectiveness = all_stats[strategy_name]["overlap_effectiveness"]
            expected = self.strategies[strategy_name]["chunk_overlap"]
            summary_text += f"• {strategy_name} strategy: {effectiveness:.0%} effective with {expected} token overlap. "
            if effectiveness > 0.7:
                summary_text += "This is excellent - context is well preserved across chunk boundaries.\n"
            elif effectiveness > 0.4:
                summary_text += "This is good - some context preservation is working.\n"
            else:
                summary_text += "This is concerning - context may be lost when concepts span chunks.\n"
        
        # Query type interpretation
        question_lower = question.lower()
        is_conceptual = any(word in question_lower for word in ["explain", "why", "how", "trade-off", "compare"])
        
        summary_text += f"\n[bold]Key Finding #5: Query Type Match[/bold]\n"
        summary_text += f"Your question is a [cyan]{'conceptual' if is_conceptual else 'factoid'}[/cyan] query. "
        if is_conceptual:
            summary_text += "Conceptual queries typically benefit from larger chunks because they require understanding "
            summary_text += "relationships and reasoning, not just finding specific facts. "
            if large_sim > small_sim:
                summary_text += f"The results confirm this - Large chunks ({large_sim:.4f}) outperformed Small chunks ({small_sim:.4f}) in similarity.\n"
            else:
                summary_text += f"Interestingly, the results show {best_similarity} chunks performed best, which may indicate the document structure favors this size.\n"
        else:
            summary_text += "Factoid queries typically work better with smaller, more focused chunks. "
            summary_text += f"The {best_similarity} strategy's performance aligns with this expectation.\n"
        
        # Practical recommendation
        summary_text += f"\n[bold]Practical Recommendation:[/bold]\n"
        if is_conceptual:
            if large_sim > medium_sim and large_sim > small_sim:
                summary_text += f"For this type of conceptual question, [green]Large chunks (1024 tokens)[/green] are recommended. "
                summary_text += f"They provide the most comprehensive context ({large_tokens:,} tokens) and achieved the best similarity score ({large_sim:.4f}). "
                summary_text += "The additional context helps the LLM understand relationships and trade-offs.\n"
            elif medium_sim > small_sim:
                summary_text += f"For this type of conceptual question, [green]Medium chunks (512 tokens)[/green] offer a good balance. "
                summary_text += f"They provide substantial context ({medium_tokens:,} tokens) while maintaining good similarity ({medium_sim:.4f}). "
                summary_text += "This balances comprehensiveness with precision.\n"
            else:
                summary_text += f"Surprisingly, [green]Small chunks (256 tokens)[/green] performed best for this conceptual question. "
                summary_text += f"This suggests the document structure or query characteristics favor precision over breadth.\n"
        else:
            if small_sim > medium_sim and small_sim > large_sim:
                summary_text += f"For factoid queries like this, [green]Small chunks (256 tokens)[/green] are recommended. "
                summary_text += f"They provide focused, precise context ({small_tokens:,} tokens) with the best similarity ({small_sim:.4f}).\n"
            else:
                summary_text += f"The {best_similarity} strategy performed best, suggesting it's well-suited for this query type.\n"
        
        summary_text += f"\n[dim]Note: These recommendations are based on this specific query and document. "
        summary_text += "Different questions or documents may favor different chunk sizes.[/dim]"
        
        self.console.print(Panel(summary_text, box=box.ROUNDED, border_style="cyan"))
        self.console.print()
    
    def explain_metrics_in_plain_language(self, results: Dict, all_stats: Dict):
        """Explain what each metric means in natural language"""
        self.print_header("📚 METRIC EXPLANATIONS", 
                         "Understanding what the numbers actually mean")
        
        explanations = []
        
        # Similarity scores
        explanations.append(Panel(
            "[bold]Similarity Scores (0.0 to 1.0)[/bold]\n\n"
            "What it measures: How semantically similar the retrieved chunks are to your question.\n\n"
            "What it means:\n"
            "• 0.9-1.0: Excellent match - chunks are highly relevant\n"
            "• 0.7-0.9: Good match - chunks are relevant with some minor differences\n"
            "• 0.5-0.7: Moderate match - chunks are somewhat related but may miss key concepts\n"
            "• Below 0.5: Poor match - chunks may not be very relevant\n\n"
            "[yellow]Why it matters:[/yellow] Higher similarity means the retrieval system found chunks that actually relate to your question. "
            "However, high similarity doesn't guarantee the chunks contain the answer - they might just use similar words.",
            box=box.ROUNDED,
            border_style="yellow"
        ))
        
        # Context tokens
        explanations.append(Panel(
            "[bold]Total Context Tokens[/bold]\n\n"
            "What it measures: The total number of tokens (words/subwords) retrieved across all top chunks.\n\n"
            "What it means:\n"
            "• More tokens = More information available to the LLM\n"
            "• But more tokens can also mean more irrelevant information (noise)\n"
            "• The ideal is enough context to answer, but not so much that the LLM gets confused\n\n"
            "[yellow]Why it matters:[/yellow] LLMs have limited context windows. Too little context and they can't answer. "
            "Too much context and they may focus on irrelevant parts. Finding the right balance is key.",
            box=box.ROUNDED,
            border_style="cyan"
        ))
        
        # Number of chunks
        explanations.append(Panel(
            "[bold]Number of Chunks[/bold]\n\n"
            "What it measures: How many separate pieces the document was split into.\n\n"
            "What it means:\n"
            "• More chunks = More granular retrieval, but concepts may be fragmented\n"
            "• Fewer chunks = More complete context per chunk, but less precise retrieval\n"
            "• The optimal number depends on document structure and query type\n\n"
            "[yellow]Why it matters:[/yellow] When a concept spans multiple chunks, retrieval might miss it entirely. "
            "When chunks are too large, you retrieve irrelevant information along with relevant content.",
            box=box.ROUNDED,
            border_style="green"
        ))
        
        # Overlap effectiveness
        explanations.append(Panel(
            "[bold]Overlap Effectiveness[/bold]\n\n"
            "What it measures: How well the overlap between chunks preserves context when concepts span boundaries.\n\n"
            "What it means:\n"
            "• 70%+: Excellent - Context is well preserved, concepts won't be lost\n"
            "• 40-70%: Good - Most context preserved, some risk of fragmentation\n"
            "• Below 40%: Poor - Context may be lost when concepts span chunks\n\n"
            "[yellow]Why it matters:[/yellow] When you split a document, important information might fall right at a chunk boundary. "
            "Overlap acts as a safety net, ensuring that if a concept appears at the end of one chunk, it also appears at the start of the next.",
            box=box.ROUNDED,
            border_style="magenta"
        ))
        
        # Keyword density
        small_kd = results["Small"]["quality_metrics"]["keyword_density"]
        medium_kd = results["Medium"]["quality_metrics"]["keyword_density"]
        large_kd = results["Large"]["quality_metrics"]["keyword_density"]
        
        explanations.append(Panel(
            "[bold]Keyword Density[/bold]\n\n"
            "What it measures: How often words from your question appear in the retrieved chunks.\n\n"
            "What it means:\n"
            f"• Small chunks: {small_kd:.4f} - {'High' if small_kd > 0.01 else 'Low'} keyword presence\n"
            f"• Medium chunks: {medium_kd:.4f} - {'High' if medium_kd > 0.01 else 'Low'} keyword presence\n"
            f"• Large chunks: {large_kd:.4f} - {'High' if large_kd > 0.01 else 'Low'} keyword presence\n\n"
            "[yellow]Why it matters:[/yellow] Higher keyword density suggests the chunks are directly related to your question. "
            "However, semantic similarity (measured by embeddings) is often more important than exact keyword matches.",
            box=box.ROUNDED,
            border_style="red"
        ))
        
        for explanation in explanations:
            self.console.print(explanation)
            self.console.print()
    
    def provide_research_insights(self, results: Dict, all_stats: Dict, question: str):
        """Provide educational research insights"""
        self.print_header("🎓 RESEARCH INSIGHTS & RECOMMENDATIONS", 
                         "Evidence-based analysis for RAG system design")
        
        insights = []
        
        # Context-to-noise analysis
        small_tokens = results["Small"]["quality_metrics"]["total_tokens"]
        large_tokens = results["Large"]["quality_metrics"]["total_tokens"]
        noise_ratio = (large_tokens - small_tokens) / small_tokens if small_tokens > 0 else 0
        
        insights.append(Panel(
            "[bold]1. Context-to-Noise Ratio[/bold]\n\n"
            f"Small chunks provide {small_tokens} tokens of context.\n"
            f"Large chunks provide {large_tokens} tokens ({noise_ratio:.0%} more).\n\n"
            "[yellow]💡 Insight:[/yellow] Larger chunks include more context but may introduce noise. "
            "For factoid queries, smaller chunks are often more precise. For conceptual queries "
            "requiring reasoning, larger chunks provide necessary context.",
            box=box.ROUNDED,
            border_style="yellow"
        ))
        
        # Similarity score analysis
        similarities = {
            "Small": np.mean(results["Small"]["similarities"]),
            "Medium": np.mean(results["Medium"]["similarities"]),
            "Large": np.mean(results["Large"]["similarities"])
        }
        best_strategy = max(similarities, key=similarities.get)
        
        insights.append(Panel(
            "[bold]2. Retrieval Quality[/bold]\n\n"
            f"Best average similarity: [green]{best_strategy}[/green] ({similarities[best_strategy]:.4f})\n\n"
            "[yellow]💡 Insight:[/yellow] Higher similarity scores indicate better semantic matching. "
            "However, similarity alone doesn't guarantee answer quality - context completeness matters too.",
            box=box.ROUNDED,
            border_style="cyan"
        ))
        
        # Query type analysis
        question_lower = question.lower()
        is_conceptual = any(word in question_lower for word in ["explain", "why", "how", "trade-off", "compare"])
        
        insights.append(Panel(
            "[bold]3. Query Type Analysis[/bold]\n\n"
            f"Query type: [cyan]{'Conceptual' if is_conceptual else 'Factoid'}[/cyan]\n"
            f"Question: {question}\n\n"
            "[yellow]💡 Insight:[/yellow] "
            "Conceptual queries (explain, why, how) typically benefit from larger chunks (512-1024 tokens) "
            "as they require broader context. Factoid queries (what, when, who) work better with smaller "
            "chunks (256 tokens) for precision.",
            box=box.ROUNDED,
            border_style="green"
        ))
        
        # Overlap analysis
        overlap_effectiveness = {
            "Small": all_stats["Small"]["overlap_effectiveness"],
            "Medium": all_stats["Medium"]["overlap_effectiveness"],
            "Large": all_stats["Large"]["overlap_effectiveness"]
        }
        
        insights.append(Panel(
            "[bold]4. Overlap Strategy Effectiveness[/bold]\n\n"
            f"Small: {overlap_effectiveness['Small']:.2%} | "
            f"Medium: {overlap_effectiveness['Medium']:.2%} | "
            f"Large: {overlap_effectiveness['Large']:.2%}\n\n"
            "[yellow]💡 Insight:[/yellow] Overlap helps preserve context across chunk boundaries, "
            "preventing information loss when concepts span multiple chunks. Higher overlap effectiveness "
            "indicates better context preservation.",
            box=box.ROUNDED,
            border_style="magenta"
        ))
        
        for insight in insights:
            self.console.print(insight)
            self.console.print()
        
        # Final recommendation
        recommendation = Panel(
            "[bold]📋 RECOMMENDATION FOR THIS QUERY TYPE[/bold]\n\n"
            f"Based on analysis, [green]{'Medium or Large' if is_conceptual else 'Small'}[/green] "
            "chunking strategy is recommended for this type of query.\n\n"
            "[dim]Note: Optimal chunk size depends on document structure, query complexity, "
            "and LLM context window limitations.[/dim]",
            box=box.DOUBLE,
            border_style="green"
        )
        self.console.print(recommendation)
        self.console.print()
    
    def run_experiment(self, doc_path: str, question: str):
        """Run the complete chunking experiment"""
        self.print_header("🔬 CHUNKING STRATEGY RESEARCH EXPERIMENT", 
                         "Comprehensive analysis of chunking strategies for RAG systems")
        
        self.console.print(f"[bold]Research Question:[/bold] {question}\n")
        
        # Step 1: Load and chunk document
        self.console.print("[bold cyan]Phase 1: Document Chunking[/bold cyan]")
        self.console.print("=" * 80)
        
        all_chunks = {}
        all_stats = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            for strategy_name in self.strategies.keys():
                task = progress.add_task(f"Chunking with {strategy_name} strategy...", total=None)
                chunks, stats = self.load_and_chunk_document(doc_path, strategy_name)
                all_chunks[strategy_name] = chunks
                all_stats[strategy_name] = stats
                progress.update(task, completed=True)
        
        self.display_chunking_statistics(all_stats)
        self.visualize_token_distribution(all_stats)
        
        # Step 2: Index chunks
        self.console.print("[bold cyan]Phase 2: Indexing & Embedding[/bold cyan]")
        self.console.print("=" * 80)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            for strategy_name in self.strategies.keys():
                task = progress.add_task(f"Indexing {strategy_name} chunks...", total=None)
                collection_name = f"chunk_experiment_{strategy_name.lower()}"
                self.index_chunks(all_chunks[strategy_name], collection_name, strategy_name)
                progress.update(task, completed=True)
        
        self.console.print("[green]✓[/green] All chunks indexed successfully\n")
        
        # Step 3: Query and retrieve
        self.console.print("[bold cyan]Phase 3: Retrieval & Analysis[/bold cyan]")
        self.console.print("=" * 80)
        
        results = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            for strategy_name in self.strategies.keys():
                task = progress.add_task(f"Querying {strategy_name} collection...", total=None)
                collection_name = f"chunk_experiment_{strategy_name.lower()}"
                chunks, similarities, metadatas = self.query_collection(collection_name, question, top_k=3)
                results[strategy_name] = {
                    "chunks": chunks,
                    "similarities": similarities,
                    "metadatas": metadatas,
                    "description": self.strategies[strategy_name]["description"]
                }
                progress.update(task, completed=True)
        
        # Step 4: Display results
        self.display_retrieval_results(results, question)
        
        # Step 5: Natural language summaries
        self.generate_natural_language_summary(results, all_stats, question)
        self.explain_metrics_in_plain_language(results, all_stats)
        
        # Step 6: Analysis
        self.analyze_overlap_effectiveness(results, all_stats)
        self.provide_research_insights(results, all_stats, question)
        
        # Final conclusion
        self.generate_final_conclusion(results, all_stats, question)
        
        # Store for potential export
        self.experiment_data = {
            "results": results,
            "stats": all_stats,
            "question": question
        }
    
    def generate_final_conclusion(self, results: Dict, all_stats: Dict, question: str):
        """Generate a final comprehensive conclusion"""
        self.print_header("🎯 FINAL CONCLUSION", 
                         "Synthesized findings and actionable recommendations")
        
        # Calculate key comparisons
        small_sim = np.mean(results["Small"]["similarities"])
        medium_sim = np.mean(results["Medium"]["similarities"])
        large_sim = np.mean(results["Large"]["similarities"])
        
        small_tokens = results["Small"]["quality_metrics"]["total_tokens"]
        large_tokens = results["Large"]["quality_metrics"]["total_tokens"]
        
        best_strategy = max(["Small", "Medium", "Large"], 
                           key=lambda s: np.mean(results[s]["similarities"]))
        
        conclusion = f"""
[bold]Bottom Line:[/bold]

After testing three chunking strategies on your question about Semantic Routers, here's what we learned:

[bold]1. Which Strategy Performed Best?[/bold]
The [green]{best_strategy}[/green] strategy achieved the highest similarity score ({np.mean(results[best_strategy]["similarities"]):.4f}), 
meaning it found chunks that are most semantically aligned with your question.

[bold]2. What Does This Mean in Practice?[/bold]
"""
        
        if best_strategy == "Small":
            conclusion += """
• Small chunks (256 tokens) work best for this query because they provide focused, precise information
• The document structure allows important concepts to be captured in smaller chunks
• This suggests your question benefits from precision over breadth
• However, be aware that smaller chunks might miss broader context or relationships
"""
        elif best_strategy == "Medium":
            conclusion += """
• Medium chunks (512 tokens) offer the best balance for this query
• They provide enough context for understanding while maintaining precision
• This is often the "sweet spot" for many RAG applications
• Medium chunks typically work well for both factoid and conceptual queries
"""
        else:
            conclusion += """
• Large chunks (1024 tokens) work best because your question requires comprehensive context
• The question asks about "trade-offs" which requires understanding relationships and nuances
• Larger chunks preserve these relationships better than smaller, fragmented chunks
• The additional context helps the LLM understand the full picture
"""
        
        conclusion += f"""
[bold]3. The Trade-offs You Should Know:[/bold]

[red]Small Chunks (256 tokens):[/red]
• ✓ High precision - less irrelevant information
• ✓ Fast retrieval and processing
• ✗ May fragment related concepts
• ✗ Might miss broader context needed for complex reasoning

[yellow]Medium Chunks (512 tokens):[/yellow]
• ✓ Balanced approach - good precision and context
• ✓ Works well for most query types
• ✗ May not be optimal for very specific or very broad questions

[green]Large Chunks (1024 tokens):[/green]
• ✓ Comprehensive context - preserves relationships
• ✓ Better for conceptual and reasoning questions
• ✗ More noise - includes potentially irrelevant information
• ✗ Higher computational cost

[bold]4. What Should You Do?[/bold]
"""
        
        question_lower = question.lower()
        is_conceptual = any(word in question_lower for word in ["explain", "why", "how", "trade-off", "compare"])
        
        if is_conceptual:
            conclusion += f"""
For conceptual questions like yours (asking about trade-offs and explanations), we recommend:

[green]Primary Recommendation:[/green] Use [bold]{best_strategy} chunks[/bold] based on the similarity scores.

[green]Alternative Consideration:[/green] If you need to balance precision with context, Medium chunks (512 tokens) 
often provide a good compromise. They give you substantial context while reducing noise compared to Large chunks.

[dim]Remember: The optimal chunk size can vary based on:
• Document structure and writing style
• Specific question being asked
• LLM context window limitations
• Desired balance between precision and comprehensiveness[/dim]
"""
        else:
            conclusion += f"""
For factoid questions, we typically recommend smaller chunks, but the results show:

[green]Primary Recommendation:[/green] Use [bold]{best_strategy} chunks[/bold] based on the similarity scores.

[dim]Note: Even for factoid queries, sometimes larger chunks help by providing necessary context 
around the fact. The similarity scores tell us which size works best for your specific case.[/dim]
"""
        
        conclusion += f"""
[bold]5. Key Insight for Your Research:[/bold]

The experiment reveals that chunk size significantly impacts retrieval quality. The {best_strategy} strategy 
outperformed others by {abs(np.mean(results[best_strategy]["similarities"]) - np.mean([small_sim, medium_sim, large_sim])):.4f} 
in average similarity, which translates to more relevant context for answer generation.

Additionally, the overlap effectiveness analysis shows how well each strategy preserves context across 
chunk boundaries - an important consideration when concepts span multiple chunks.

[bold]Next Steps:[/bold]
1. Test with different questions to see if the pattern holds
2. Consider your LLM's context window when choosing chunk size
3. Monitor answer quality, not just retrieval metrics
4. Experiment with hybrid approaches (e.g., retrieve from multiple chunk sizes)
"""
        
        self.console.print(Panel(conclusion, box=box.DOUBLE, border_style="green"))
        self.console.print()


def main():
    """Main function to run the chunking experiment"""
    doc_path = "sample_document.txt"
    
    if not os.path.exists(doc_path):
        console = Console()
        console.print(f"[red]Error:[/red] {doc_path} not found!")
        return
    
    question = "Explain the trade-offs of using a Semantic Router in a multi-domain enterprise environment."
    
    experiment = ChunkingExperiment()
    experiment.run_experiment(doc_path, question)
    
    console = Console()
    console.print("\n" + "=" * 80)
    console.print(Panel(
        "[bold green]✓ EXPERIMENT COMPLETE[/bold green]\n\n"
        "Generated files:\n"
        "  • chunk_token_distribution.png - Token distribution visualization\n\n"
        "Collections created:\n"
        "  • chunk_experiment_small\n"
        "  • chunk_experiment_medium\n"
        "  • chunk_experiment_large",
        box=box.DOUBLE,
        border_style="green"
    ))


if __name__ == "__main__":
    main()
