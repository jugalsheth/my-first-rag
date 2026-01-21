"""
Multi-Query RAG System
Implements query expansion using Gemini to generate multiple query variations,
then combines and deduplicates results for better coverage.
"""

import os
import warnings
import time
import sys
from typing import List, Dict, Tuple, Set
from collections import OrderedDict

# Suppress harmless urllib3/OpenSSL warning on macOS
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')

# Show loading message early
print("🔍 Multi-Query RAG System")
print("Loading dependencies (this may take 10-30 seconds on first run)...")
sys.stdout.flush()

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

print("  ✓ Environment loaded")
sys.stdout.flush()

print("  Loading ChromaDB...")
sys.stdout.flush()
import chromadb
from chromadb.config import Settings

print("  Loading sentence-transformers (this may take a moment)...")
sys.stdout.flush()
from sentence_transformers import SentenceTransformer

print("  Loading Gemini API...")
sys.stdout.flush()
import google.generativeai as genai

print("  Loading Rich for beautiful output...")
sys.stdout.flush()
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from rich.layout import Layout

print("  ✓ All dependencies loaded\n")
sys.stdout.flush()


class MultiQueryRAG:
    def __init__(self, collection_name: str = "chunk_experiment_small", 
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Multi-Query RAG system
        
        Args:
            collection_name: Name of the ChromaDB collection to use
            embedding_model_name: Embedding model for retrieval
        """
        self.console = Console()
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load embedding model
        with self.console.status("[bold green]Loading embedding model...") as status:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize Gemini for query expansion
        self.gemini_client = None
        self._init_gemini()
        
        # Get ChromaDB collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            self.console.print(f"[green]✓[/green] Loaded collection: [bold]{collection_name}[/bold]")
        except Exception as e:
            self.console.print(f"[red]Error: Collection '{collection_name}' not found![/red]")
            self.console.print("[yellow]Available collections:[/yellow]")
            try:
                collections = self.client.list_collections()
                for col in collections:
                    self.console.print(f"  • {col.name}")
            except:
                pass
            raise e
    
    def _init_gemini(self):
        """Initialize Gemini client for query expansion"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.console.print("[red]Error: GEMINI_API_KEY environment variable not set[/red]")
            self.console.print("[yellow]Set it in .env file or export it[/yellow]")
            raise Exception("GEMINI_API_KEY not found")
        
        genai.configure(api_key=api_key)
        
        # Try models with higher quotas first
        models_to_try = [
            ('models/gemini-2.5-flash-lite', 10, "Gemini 2.5 Flash Lite"),
            ('models/gemma-3-12b-it', 30, "Gemma 3 12B"),
            ('models/gemma-3-4b-it', 30, "Gemma 3 4B"),
            ('models/gemini-2.5-flash', 5, "Gemini 2.5 Flash")
        ]
        
        for model_name, rpm, display_name in models_to_try:
            try:
                self.gemini_client = genai.GenerativeModel(model_name)
                self.rpm_limit = rpm
                self.model_name = model_name
                self.console.print(f"[green]✓[/green] Using {display_name} for query expansion ({rpm} RPM)")
                return
            except Exception as e:
                continue
        
        raise Exception("No Gemini models available")
    
    def print_header(self, title: str, subtitle: str = ""):
        """Print a beautiful header"""
        header_text = f"[bold cyan]{title}[/bold cyan]"
        if subtitle:
            header_text += f"\n[dim]{subtitle}[/dim]"
        self.console.print(Panel(header_text, box=box.DOUBLE, expand=False))
        self.console.print()
    
    def generate_query_variations(self, original_query: str, num_variations: int = 3) -> List[str]:
        """
        Generate query variations using Gemini
        
        Args:
            original_query: The original user question
            num_variations: Number of query variations to generate (default: 3)
            
        Returns:
            List of query variations including the original
        """
        self.console.print(f"[bold]Generating {num_variations} query variations...[/bold]")
        
        prompt = f"""You are a query expansion system for a RAG (Retrieval-Augmented Generation) system.

Given the following user question, generate {num_variations} alternative phrasings or reformulations that would help retrieve more comprehensive information. Each variation should:
- Use different wording or phrasing
- Focus on different aspects or angles of the question
- Use synonyms or related terms
- Be concise and searchable (1-2 sentences max)

Original question: {original_query}

Generate exactly {num_variations} variations. Return ONLY the variations, one per line, without numbering or bullets.

Example format:
variation 1
variation 2
variation 3"""

        try:
            # Rate limiting: wait if needed (10 RPM = 6 seconds between calls)
            delay = 60 / self.rpm_limit
            if delay < 6:
                delay = 6  # Safe buffer
            
            response = self.gemini_client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,  # More creative for variations
                    "max_output_tokens": 300,
                }
            )
            
            # Parse variations (one per line)
            variations_text = response.text.strip()
            variations = [v.strip() for v in variations_text.split('\n') if v.strip()]
            
            # Clean up variations (remove numbering if present)
            cleaned_variations = []
            for var in variations:
                # Remove leading numbers, bullets, dashes
                var = var.lstrip('0123456789.-• ').strip()
                if var:
                    cleaned_variations.append(var)
            
            # Ensure we have exactly num_variations
            if len(cleaned_variations) > num_variations:
                cleaned_variations = cleaned_variations[:num_variations]
            elif len(cleaned_variations) < num_variations:
                # If we got fewer, pad with simple variations
                while len(cleaned_variations) < num_variations:
                    cleaned_variations.append(original_query)
            
            # Combine original with variations
            all_queries = [original_query] + cleaned_variations[:num_variations]
            
            self.console.print(f"[green]✓[/green] Generated {len(cleaned_variations)} variations")
            return all_queries
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                self.console.print(f"[red]Rate limit error: {error_msg}[/red]")
                self.console.print("[yellow]Falling back to template-based variations...[/yellow]")
                # Fallback: simple template-based variations
                return self._template_query_variations(original_query, num_variations)
            else:
                self.console.print(f"[yellow]Gemini error: {e}[/yellow]")
                self.console.print("[yellow]Falling back to template-based variations...[/yellow]")
                return self._template_query_variations(original_query, num_variations)
    
    def _template_query_variations(self, query: str, num_variations: int) -> List[str]:
        """Fallback: generate simple template-based variations"""
        # Simple keyword extraction and rephrasing
        words = query.lower().split()
        
        variations = [query]  # Original as first
        
        # Variation 1: Add "what are" prefix if not present
        if not query.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            variations.append(f"What are {query.lower()}?")
        
        # Variation 2: Add "explain" prefix
        if not query.lower().startswith('explain'):
            variations.append(f"Explain {query.lower()}")
        
        # Variation 3: Question form
        if '?' not in query:
            variations.append(f"{query}?")
        else:
            variations.append(query.replace('?', ''))
        
        return variations[:num_variations + 1]  # +1 to include original
    
    def retrieve_chunks(self, query: str, top_k: int = 3) -> Tuple[List[str], List[float], List[str]]:
        """
        Retrieve chunks from ChromaDB for a query
        
        Returns:
            Tuple of (chunks, similarities, chunk_ids)
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Extract results
        chunks = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        ids = results['ids'][0] if results['ids'] else []
        
        # Convert distances to similarities (1 - distance for cosine)
        similarities = [1 - d for d in distances]
        
        return chunks, similarities, ids
    
    def deduplicate_chunks(self, all_chunks: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        """
        Deduplicate chunks by ID, keeping the highest similarity score
        
        Args:
            all_chunks: List of (chunk_text, similarity, chunk_id) tuples
            
        Returns:
            Deduplicated list, sorted by similarity (descending)
        """
        # Use OrderedDict to preserve order while deduplicating
        seen = {}
        
        for chunk_text, similarity, chunk_id in all_chunks:
            if chunk_id not in seen:
                seen[chunk_id] = (chunk_text, similarity, chunk_id)
            else:
                # Keep the one with higher similarity
                if similarity > seen[chunk_id][1]:
                    seen[chunk_id] = (chunk_text, similarity, chunk_id)
        
        # Sort by similarity (descending)
        deduplicated = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        
        return deduplicated
    
    def calculate_coverage(self, single_query_results: List[str], 
                          multi_query_results: List[str]) -> Dict:
        """
        Calculate coverage improvement metrics
        
        Returns:
            Dictionary with coverage metrics
        """
        single_query_set = set(single_query_results)
        multi_query_set = set(multi_query_results)
        
        # Unique chunks found by each
        single_unique = len(single_query_set)
        multi_unique = len(multi_query_set)
        
        # Coverage improvement
        if single_unique > 0:
            coverage_improvement = ((multi_unique - single_unique) / single_unique) * 100
        else:
            coverage_improvement = 0.0
        
        # Overlap analysis
        overlap = single_query_set & multi_query_set
        overlap_count = len(overlap)
        
        # New chunks found by multi-query
        new_chunks = multi_query_set - single_query_set
        new_count = len(new_chunks)
        
        return {
            "single_query_count": single_unique,
            "multi_query_count": multi_unique,
            "coverage_improvement": coverage_improvement,
            "overlap_count": overlap_count,
            "new_chunks_count": new_count,
            "overlap_chunks": list(overlap),
            "new_chunks": list(new_chunks)
        }
    
    def run_comparison(self, question: str, top_k_per_query: int = 3, 
                      num_variations: int = 3) -> Dict:
        """
        Run the complete multi-query comparison
        
        Args:
            question: The user question
            top_k_per_query: Number of chunks to retrieve per query
            num_variations: Number of query variations to generate
            
        Returns:
            Dictionary with all results
        """
        self.print_header("🔍 MULTI-QUERY RAG COMPARISON", 
                         f"Testing: '{question}'")
        
        # Step 1: Generate query variations
        queries = self.generate_query_variations(question, num_variations)
        
        # Display generated queries
        self.console.print()
        query_table = Table(title="Generated Queries", show_header=True, 
                           header_style="bold magenta", box=box.ROUNDED)
        query_table.add_column("Query #", style="cyan", justify="right")
        query_table.add_column("Query Text", style="yellow")
        
        query_table.add_row("Original", queries[0])
        for i, query in enumerate(queries[1:], 1):
            query_table.add_row(f"Variation {i}", query)
        
        self.console.print(query_table)
        self.console.print()
        
        # Step 2: Retrieve chunks for each query
        self.console.print("[bold]Retrieving chunks for each query...[/bold]")
        
        all_results = {}
        all_chunks_with_ids = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            for i, query in enumerate(queries):
                query_label = "Original" if i == 0 else f"Variation {i}"
                task = progress.add_task(f"Querying: {query[:50]}...", total=None)
                
                chunks, similarities, ids = self.retrieve_chunks(query, top_k_per_query)
                
                all_results[query_label] = {
                    "query": query,
                    "chunks": chunks,
                    "similarities": similarities,
                    "ids": ids
                }
                
                # Collect all chunks with IDs for deduplication
                for chunk, sim, chunk_id in zip(chunks, similarities, ids):
                    all_chunks_with_ids.append((chunk, sim, chunk_id))
                
                progress.update(task, completed=True)
        
        # Step 3: Get single-query baseline (original query only)
        single_query_chunks = all_results["Original"]["chunks"]
        single_query_ids = all_results["Original"]["ids"]
        
        # Step 4: Combine and deduplicate multi-query results
        self.console.print()
        self.console.print("[bold]Combining and deduplicating results...[/bold]")
        
        deduplicated_chunks = self.deduplicate_chunks(all_chunks_with_ids)
        multi_query_chunks = [chunk for chunk, _, _ in deduplicated_chunks]
        multi_query_ids = [chunk_id for _, _, chunk_id in deduplicated_chunks]
        
        # Step 5: Calculate coverage metrics
        coverage_metrics = self.calculate_coverage(
            single_query_ids,
            multi_query_ids
        )
        
        # Step 6: Display results
        self.display_results(all_results, single_query_chunks, multi_query_chunks, 
                           deduplicated_chunks, coverage_metrics, question, top_k_per_query)
        
        return {
            "question": question,
            "queries": queries,
            "single_query_results": {
                "chunks": single_query_chunks,
                "ids": single_query_ids
            },
            "multi_query_results": {
                "chunks": multi_query_chunks,
                "ids": multi_query_ids
            },
            "coverage_metrics": coverage_metrics,
            "all_query_results": all_results
        }
    
    def display_results(self, all_results: Dict, single_query_chunks: List[str],
                       multi_query_chunks: List[str], deduplicated_chunks: List[Tuple],
                       coverage_metrics: Dict, original_question: str, top_k_per_query: int):
        """Display comprehensive comparison results"""
        
        # Coverage Summary
        self.print_header("📊 COVERAGE ANALYSIS", "Single Query vs Multi-Query")
        
        coverage_table = Table(show_header=True, header_style="bold green", box=box.ROUNDED)
        coverage_table.add_column("Metric", style="cyan")
        coverage_table.add_column("Single Query", justify="right", style="yellow")
        coverage_table.add_column("Multi-Query", justify="right", style="green")
        coverage_table.add_column("Improvement", justify="right", style="bold")
        
        coverage_table.add_row(
            "Unique Chunks Found",
            str(coverage_metrics["single_query_count"]),
            str(coverage_metrics["multi_query_count"]),
            f"[green]+{coverage_metrics['new_chunks_count']} ({coverage_metrics['coverage_improvement']:.1f}%)[/green]"
        )
        
        coverage_table.add_row(
            "Overlapping Chunks",
            str(coverage_metrics["overlap_count"]),
            str(coverage_metrics["overlap_count"]),
            "[dim]Same[/dim]"
        )
        
        coverage_table.add_row(
            "New Chunks (Multi-Query Only)",
            "[dim]0[/dim]",
            str(coverage_metrics["new_chunks_count"]),
            f"[green]+{coverage_metrics['new_chunks_count']}[/green]"
        )
        
        self.console.print(coverage_table)
        self.console.print()
        
        # Interpretation
        improvement = coverage_metrics["coverage_improvement"]
        if improvement > 50:
            interpretation = "[green]Excellent improvement![/green] Multi-query found significantly more unique chunks."
        elif improvement > 20:
            interpretation = "[yellow]Good improvement.[/yellow] Multi-query found additional relevant chunks."
        elif improvement > 0:
            interpretation = "[yellow]Moderate improvement.[/yellow] Multi-query found some additional chunks."
        else:
            interpretation = "[red]No improvement.[/red] Multi-query didn't find additional unique chunks."
        
        self.console.print(Panel(
            f"[bold]Interpretation:[/bold]\n\n{interpretation}\n\n"
            f"Multi-query expanded coverage by [green]{improvement:.1f}%[/green], finding "
            f"[green]{coverage_metrics['new_chunks_count']}[/green] additional unique chunks that "
            f"the single query missed.",
            box=box.ROUNDED,
            border_style="cyan"
        ))
        self.console.print()
        
        # Detailed chunk comparison
        self.print_header("📄 RETRIEVED CHUNKS COMPARISON", 
                         "Side-by-side view of retrieved content")
        
        # Single Query Results
        self.console.print(Panel(
            f"[bold yellow]Single Query (Original)[/bold yellow]\n"
            f"[dim]Query: {all_results['Original']['query']}[/dim]\n\n"
            f"Found [yellow]{len(single_query_chunks)}[/yellow] unique chunks:",
            box=box.ROUNDED,
            border_style="yellow"
        ))
        
        for i, (chunk, similarity, chunk_id) in enumerate(zip(
            all_results["Original"]["chunks"],
            all_results["Original"]["similarities"],
            all_results["Original"]["ids"]
        ), 1):
            display_chunk = chunk[:300] + "..." if len(chunk) > 300 else chunk
            self.console.print(Panel(
                f"[bold]Chunk {i}[/bold] (ID: [dim]{chunk_id[:30]}...[/dim])\n"
                f"Similarity: [cyan]{similarity:.4f}[/cyan]\n\n"
                f"{display_chunk}",
                box=box.SIMPLE,
                border_style="yellow"
            ))
        
        self.console.print()
        
        # Multi-Query Results (Top chunks after deduplication)
        self.console.print(Panel(
            f"[bold green]Multi-Query (Combined & Deduplicated)[/bold green]\n"
            f"[dim]Used {len(all_results)} queries: Original + {len(all_results)-1} variations[/dim]\n\n"
            f"Found [green]{len(multi_query_chunks)}[/green] unique chunks after deduplication:",
            box=box.ROUNDED,
            border_style="green"
        ))
        
        # Show top chunks (up to top_k * 2 to show the benefit)
        max_display = min(len(multi_query_chunks), top_k_per_query * 2)
        for i, (chunk, similarity, chunk_id) in enumerate(deduplicated_chunks[:max_display], 1):
            # Check if this is a "new" chunk (not in single query)
            is_new = chunk_id not in all_results["Original"]["ids"]
            new_badge = "[green][NEW][/green] " if is_new else ""
            
            display_chunk = chunk[:300] + "..." if len(chunk) > 300 else chunk
            self.console.print(Panel(
                f"{new_badge}[bold]Chunk {i}[/bold] (ID: [dim]{chunk_id[:30]}...[/dim])\n"
                f"Similarity: [cyan]{similarity:.4f}[/cyan]\n\n"
                f"{display_chunk}",
                box=box.SIMPLE,
                border_style="green" if is_new else "dim"
            ))
        
        self.console.print()
        
        # Query-by-Query Breakdown
        self.print_header("🔎 QUERY-BY-QUERY BREAKDOWN", 
                         "See what each query variation found")
        
        for query_label, result in all_results.items():
            color = "yellow" if query_label == "Original" else "cyan"
            
            self.console.print(Panel(
                f"[bold {color}]{query_label}[/bold {color}]\n"
                f"[dim]{result['query']}[/dim]\n\n"
                f"Found [yellow]{len(result['chunks'])}[/yellow] chunks:",
                box=box.ROUNDED,
                border_style=color
            ))
            
            for i, (chunk, similarity, chunk_id) in enumerate(zip(
                result["chunks"],
                result["similarities"],
                result["ids"]
            ), 1):
                display_chunk = chunk[:200] + "..." if len(chunk) > 200 else chunk
                self.console.print(f"  {i}. [dim]({chunk_id[:20]}...)[/dim] Similarity: [cyan]{similarity:.4f}[/cyan]")
                self.console.print(f"     {display_chunk[:150]}...")
                self.console.print()
        
        # Final Summary
        self.print_header("🎯 KEY INSIGHTS", "What the data tells us")
        
        insights = []
        
        # Insight 1: Coverage improvement
        if coverage_metrics["coverage_improvement"] > 0:
            insights.append(
                f"✓ [green]Multi-query improved coverage by {coverage_metrics['coverage_improvement']:.1f}%[/green]\n"
                f"  Found {coverage_metrics['new_chunks_count']} additional unique chunks"
            )
        else:
            insights.append(
                f"⚠ [yellow]Multi-query did not improve coverage[/yellow]\n"
                f"  All queries retrieved similar chunks (query variations may be too similar)"
            )
        
        # Insight 2: Query diversity
        unique_chunks_per_query = set()
        for result in all_results.values():
            unique_chunks_per_query.update(result["ids"])
        
        if len(unique_chunks_per_query) > coverage_metrics["single_query_count"]:
            insights.append(
                f"✓ [green]Query variations found diverse content[/green]\n"
                f"  Different phrasings retrieved different chunks"
            )
        else:
            insights.append(
                f"⚠ [yellow]Query variations found similar content[/yellow]\n"
                f"  Variations may need more diverse phrasing"
            )
        
        # Insight 3: Token cost trade-off
        single_tokens = sum(len(chunk.split()) for chunk in single_query_chunks)
        multi_tokens = sum(len(chunk.split()) for chunk in multi_query_chunks[:len(single_query_chunks)])
        token_increase = ((len(multi_query_chunks) - len(single_query_chunks)) / len(single_query_chunks) * 100) if single_query_chunks else 0
        
        insights.append(
            f"📊 [cyan]Token cost analysis:[/cyan]\n"
            f"  Single query: ~{single_tokens} tokens\n"
            f"  Multi-query: ~{multi_tokens * len(all_results)} tokens ({len(all_results)}x queries)\n"
            f"  Trade-off: {len(all_results)}x API calls for {coverage_metrics['coverage_improvement']:.1f}% more coverage"
        )
        
        summary_text = "\n\n".join(insights)
        
        self.console.print(Panel(
            summary_text,
            box=box.ROUNDED,
            border_style="green"
        ))
        self.console.print()


def main():
    """Main function to run Multi-Query RAG comparison"""
    import sys
    
    # Default question (can be overridden via command line)
    question = "What are the trade-offs of using small vs large chunks in RAG?"
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    
    # Default collection (can be changed)
    collection_name = "chunk_experiment_small"
    
    # Check if user wants to use a different collection
    if "--collection" in sys.argv or "-c" in sys.argv:
        try:
            idx = sys.argv.index("--collection" if "--collection" in sys.argv else "-c")
            collection_name = sys.argv[idx + 1]
        except (IndexError, ValueError):
            pass
    
    # Check available collections
    try:
        client = chromadb.PersistentClient(path="./chroma_db", settings=Settings(anonymized_telemetry=False))
        collections = client.list_collections()
        collection_names = [col.name for col in collections]
        
        console = Console()
        if collection_name not in collection_names:
            console.print(f"[yellow]Warning: Collection '{collection_name}' not found.[/yellow]")
            console.print("[yellow]Available collections:[/yellow]")
            for col_name in collection_names:
                console.print(f"  • {col_name}")
            if collection_names:
                collection_name = collection_names[0]
                console.print(f"[green]Using: {collection_name}[/green]\n")
            else:
                console.print("[red]No collections found! Run chunk_experiment.py first.[/red]")
                return
    except Exception as e:
        console = Console()
        console.print(f"[yellow]Could not check collections: {e}[/yellow]")
    
    # Run the comparison
    try:
        multi_query_rag = MultiQueryRAG(collection_name=collection_name)
        results = multi_query_rag.run_comparison(
            question=question,
            top_k_per_query=3,
            num_variations=3
        )
        
        # Print final summary
        console = Console()
        console.print()
        console.print("=" * 80)
        console.print(Panel(
            "[bold green]✓ MULTI-QUERY RAG COMPARISON COMPLETE[/bold green]\n\n"
            f"[bold]Question:[/bold] {question}\n\n"
            f"[bold]Results:[/bold]\n"
            f"  • Single Query: {results['coverage_metrics']['single_query_count']} unique chunks\n"
            f"  • Multi-Query: {results['coverage_metrics']['multi_query_count']} unique chunks\n"
            f"  • Coverage Improvement: [green]{results['coverage_metrics']['coverage_improvement']:.1f}%[/green]\n"
            f"  • New Chunks Found: [green]{results['coverage_metrics']['new_chunks_count']}[/green]",
            box=box.DOUBLE,
            border_style="green"
        ))
        
    except Exception as e:
        console = Console()
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    main()
