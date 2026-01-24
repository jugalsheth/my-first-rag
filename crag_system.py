"""
CRAG (Corrective RAG) System with Web Search Fallback
Extends Self-RAG with intelligent routing to web search when local retrieval is insufficient.

Flow:
1. Retrieve top-k chunks from local ChromaDB
2. Use Gemini to score relevance (1-5 scale) for each chunk
3. Calculate average relevance score
4. Route based on score:
   - 4.0-5.0: Use local docs only (high confidence)
   - 3.0-3.9: Hybrid mode (combine local + web)
   - 2.0-2.9: Use web search only (low confidence)
   - 0.0-1.9: Decline to answer ("I don't know")
5. Generate answer from selected source(s)

This proves the system can intelligently fallback to web search when local knowledge is insufficient.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

# Early progress messages
print("🌐 CRAG System (Corrective RAG with Web Search)")
print("Loading dependencies...")
sys.stdout.flush()

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except Exception:
    TAVILY_AVAILABLE = False

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box


class CRAG:
    """CRAG system with web search fallback"""
    
    def __init__(
        self,
        collection_name: str = "chunk_experiment_small",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize CRAG system
        
        Args:
            collection_name: ChromaDB collection to use
            embedding_model_name: Embedding model for retrieval
        """
        self.console = Console()
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load embedding model
        with self.console.status("[bold green]Loading embedding model...") as status:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize Gemini for judging and answering
        self.gemini_client = None
        self._init_gemini()
        
        # Initialize Tavily for web search
        self.tavily_client = None
        self._init_tavily()
        
        # Get collection
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
        """Initialize Gemini client"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.console.print("[yellow]Warning: GEMINI_API_KEY not set. Will use template-based judging.[/yellow]")
            return
        
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
                self.console.print(f"[green]✓[/green] Using {display_name} for judging ({rpm} RPM)")
                return
            except Exception:
                continue
        
        self.console.print("[yellow]Warning: No Gemini models available. Using template-based judging.[/yellow]")
    
    def _init_tavily(self):
        """Initialize Tavily client for web search"""
        if not TAVILY_AVAILABLE:
            self.console.print("[yellow]Warning: tavily-python not installed. Web search will be unavailable.[/yellow]")
            self.console.print("[dim]Install with: pip3 install tavily-python[/dim]")
            return
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            self.console.print("[yellow]Warning: TAVILY_API_KEY not set. Web search will be unavailable.[/yellow]")
            self.console.print("[dim]Get free API key at: https://tavily.com[/dim]")
            return
        
        try:
            self.tavily_client = TavilyClient(api_key=api_key)
            self.console.print("[green]✓[/green] Tavily web search enabled")
        except Exception as e:
            self.console.print(f"[yellow]Warning: Failed to initialize Tavily: {e}[/yellow]")
    
    def retrieve_chunks(self, question: str, top_k: int = 3) -> Tuple[List[str], List[float], List[str]]:
        """Retrieve chunks from ChromaDB"""
        query_embedding = self.embedding_model.encode([question]).tolist()[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        chunks = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        ids = results['ids'][0] if results['ids'] else []
        
        similarities = [1 - d for d in distances]
        
        return chunks, similarities, ids
    
    def judge_relevance(self, question: str, chunks: List[str]) -> List[float]:
        """
        Use Gemini as JUDGE to score relevance of each chunk (1-5 scale)
        
        Returns:
            List of relevance scores (1.0 to 5.0) for each chunk
        """
        scores = []
        
        if not self.gemini_client:
            # Template-based fallback: simple keyword matching
            question_words = set(question.lower().split())
            for chunk in chunks:
                chunk_words = set(chunk.lower().split())
                overlap = len(question_words & chunk_words) / len(question_words) if question_words else 0
                # Map overlap to 1-5 scale
                score = 1.0 + (overlap * 4.0)
                scores.append(min(5.0, max(1.0, score)))
            return scores
        
        # Use Gemini to judge each chunk
        for i, chunk in enumerate(chunks):
            prompt = f"""You are a relevance judge for a RAG system. Rate how relevant this chunk is to answering the question.

Question: {question}

Chunk:
{chunk[:500]}

Rate the relevance on a scale of 1-5:
- 1: Not relevant at all
- 2: Slightly relevant but doesn't help answer
- 3: Moderately relevant, partially helpful
- 4: Highly relevant, very helpful
- 5: Perfectly relevant, directly answers the question

Respond with ONLY a single number (1, 2, 3, 4, or 5). No explanation."""
            
            try:
                # Rate limiting
                if i > 0:
                    delay = max(6, 60 / self.rpm_limit)
                    time.sleep(delay)
                
                response = self.gemini_client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1,  # Low temperature for consistent judging
                        "max_output_tokens": 10,
                    }
                )
                
                # Parse score
                score_text = response.text.strip()
                # Extract first number
                import re
                numbers = re.findall(r'\d+', score_text)
                if numbers:
                    score = float(numbers[0])
                    scores.append(max(1.0, min(5.0, score)))  # Clamp to 1-5
                else:
                    scores.append(3.0)  # Default if parsing fails
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    self.console.print(f"[yellow]Rate limit hit for chunk {i+1}, using fallback scoring[/yellow]")
                    # Fallback to template-based
                    question_words = set(question.lower().split())
                    chunk_words = set(chunk.lower().split())
                    overlap = len(question_words & chunk_words) / len(question_words) if question_words else 0
                    score = 1.0 + (overlap * 4.0)
                    scores.append(min(5.0, max(1.0, score)))
                else:
                    self.console.print(f"[yellow]Error judging chunk {i+1}: {e}, using default score 3.0[/yellow]")
                    scores.append(3.0)
        
        return scores
    
    def search_web(self, question: str, max_results: int = 5) -> List[Dict]:
        """
        Search the web using Tavily API
        
        Returns:
            List of search results with 'content', 'url', 'title' keys
        """
        if not self.tavily_client:
            return []
        
        try:
            response = self.tavily_client.search(
                query=question,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
                include_raw_content=False
            )
            
            results = []
            # Tavily returns results in 'results' key
            if 'results' in response:
                for result in response['results']:
                    results.append({
                        'title': result.get('title', 'No title'),
                        'url': result.get('url', ''),
                        'content': result.get('content', '')
                    })
            
            # Also check for 'answer' in response
            if 'answer' in response and response['answer']:
                results.insert(0, {
                    'title': 'Tavily Answer',
                    'url': '',
                    'content': response['answer']
                })
            
            return results
        except Exception as e:
            self.console.print(f"[yellow]Web search error: {e}[/yellow]")
            return []
    
    def determine_routing(self, average_score: float) -> Tuple[str, str]:
        """
        Determine routing decision based on relevance score
        
        Returns:
            (source_type, reasoning) tuple
            source_type: 'local', 'web', 'hybrid', 'decline'
        """
        if average_score >= 4.0:
            return ("local", f"High confidence (score {average_score:.2f}) - using local docs only")
        elif average_score >= 3.0:
            return ("hybrid", f"Medium confidence (score {average_score:.2f}) - combining local + web")
        elif average_score >= 2.0:
            return ("web", f"Low confidence (score {average_score:.2f}) - using web search only")
        else:
            return ("decline", f"Very low confidence (score {average_score:.2f}) - declining to answer")
    
    def generate_answer(self, question: str, local_chunks: List[str] = None, web_results: List[Dict] = None) -> str:
        """Generate answer from local chunks and/or web results"""
        if not self.gemini_client:
            # Template fallback
            if local_chunks:
                context = "\n\n".join(local_chunks[:2])
                return f"Based on the retrieved information: {context[:300]}..."
            elif web_results:
                return f"Based on web search: {web_results[0].get('content', '')[:300]}..."
            return "I don't have enough information to answer this question."
        
        # Build context from sources
        context_parts = []
        
        if local_chunks:
            context_parts.append("=== LOCAL KNOWLEDGE BASE ===\n" + "\n\n".join(local_chunks))
        
        if web_results:
            web_content = "\n\n".join([
                f"Source: {r.get('title', 'Unknown')} ({r.get('url', '')})\n{r.get('content', '')}"
                for r in web_results[:3]
            ])
            context_parts.append("=== WEB SEARCH RESULTS ===\n" + web_content)
        
        if not context_parts:
            return "I don't have enough information to answer this question."
        
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""Based on the following context from local knowledge base and/or web search, answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer this question."

{context_text}

Question: {question}

Answer:"""
        
        try:
            response = self.gemini_client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 500,
                }
            )
            return response.text.strip()
        except Exception as e:
            self.console.print(f"[yellow]Error generating answer: {e}, using template[/yellow]")
            if local_chunks:
                context = "\n\n".join(local_chunks[:2])
                return f"Based on the retrieved information: {context[:300]}..."
            elif web_results:
                return f"Based on web search: {web_results[0].get('content', '')[:300]}..."
            return "I don't have enough information to answer this question."
    
    def process_question(
        self,
        question: str,
        top_k: int = 3,
        show_details: bool = True
    ) -> Dict:
        """
        Process a question through CRAG pipeline
        
        Returns:
            Dictionary with results including decision, scores, answer, sources, etc.
        """
        if show_details:
            self.console.print()
            self.console.print(Panel(
                f"[bold]Question:[/bold] {question}",
                box=box.ROUNDED,
                border_style="cyan"
            ))
        
        # Step 1: Retrieve chunks from local ChromaDB
        local_chunks, similarities, ids = self.retrieve_chunks(question, top_k)
        
        if not local_chunks:
            # No local chunks, try web search
            if show_details:
                self.console.print("[dim]No local chunks found. Searching web...[/dim]")
            
            web_results = self.search_web(question)
            if web_results:
                answer = self.generate_answer(question, local_chunks=None, web_results=web_results)
                return {
                    "question": question,
                    "source": "web",
                    "reasoning": "No local chunks found, using web search",
                    "local_chunks": [],
                    "local_scores": [],
                    "average_score": 0.0,
                    "web_results": web_results,
                    "answer": answer
                }
            else:
                return {
                    "question": question,
                    "source": "decline",
                    "reasoning": "No local chunks and web search unavailable",
                    "local_chunks": [],
                    "local_scores": [],
                    "average_score": 0.0,
                    "web_results": [],
                    "answer": "I don't have enough information to answer this question."
                }
        
        # Step 2: Judge relevance of local chunks
        if show_details:
            self.console.print("[dim]Judging relevance of retrieved chunks...[/dim]")
        
        relevance_scores = self.judge_relevance(question, local_chunks)
        average_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        # Step 3: Determine routing
        source_type, reasoning = self.determine_routing(average_score)
        
        # Step 4: Execute routing decision
        web_results = []
        final_local_chunks = []
        
        if source_type == "local":
            # Use local only
            final_local_chunks = local_chunks
            if show_details:
                self.console.print("[dim]Using local knowledge base only...[/dim]")
        
        elif source_type == "web":
            # Use web only
            if show_details:
                self.console.print("[dim]Local relevance too low. Searching web...[/dim]")
            web_results = self.search_web(question)
            if not web_results:
                # Web search failed, fallback to decline
                source_type = "decline"
                reasoning = "Web search unavailable, declining to answer"
        
        elif source_type == "hybrid":
            # Combine local + web
            final_local_chunks = local_chunks
            if show_details:
                self.console.print("[dim]Combining local knowledge with web search...[/dim]")
            web_results = self.search_web(question)
        
        elif source_type == "decline":
            # Decline to answer
            pass
        
        # Step 5: Generate answer
        if source_type == "decline":
            answer = "I don't have enough information to answer this question."
        else:
            if show_details:
                self.console.print("[dim]Generating answer...[/dim]")
            answer = self.generate_answer(
                question,
                local_chunks=final_local_chunks if final_local_chunks else None,
                web_results=web_results if web_results else None
            )
        
        return {
            "question": question,
            "source": source_type,
            "reasoning": reasoning,
            "local_chunks": local_chunks,
            "similarities": similarities,
            "ids": ids,
            "relevance_scores": relevance_scores,
            "average_score": average_score,
            "web_results": web_results,
            "answer": answer
        }
    
    def display_results(self, result: Dict):
        """Display CRAG results beautifully"""
        self.console.print()
        self.console.print(Panel(
            "[bold]CRAG RESULTS[/bold]",
            box=box.DOUBLE,
            border_style="cyan"
        ))
        self.console.print()
        
        # Local chunks with scores (if any)
        if result["local_chunks"]:
            chunks_table = Table(
                title="Local Retrieved Chunks with Relevance Scores",
                show_header=True,
                header_style="bold magenta",
                box=box.ROUNDED
            )
            
            chunks_table.add_column("Chunk #", style="cyan", justify="right")
            chunks_table.add_column("Similarity", justify="right", style="yellow")
            chunks_table.add_column("Relevance Score", justify="right", style="green")
            chunks_table.add_column("Preview", style="dim")
            
            for i, (chunk, similarity, score) in enumerate(zip(
                result["local_chunks"],
                result["similarities"],
                result["relevance_scores"]
            ), 1):
                # Color code relevance score
                if score >= 4.0:
                    score_color = "[green]"
                elif score >= 3.0:
                    score_color = "[yellow]"
                else:
                    score_color = "[red]"
                
                preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                chunks_table.add_row(
                    str(i),
                    f"{similarity:.4f}",
                    f"{score_color}{score:.1f}/5.0[/{score_color}]",
                    preview
                )
            
            self.console.print(chunks_table)
            self.console.print()
        
        # Web results (if any)
        if result["web_results"]:
            web_table = Table(
                title="Web Search Results",
                show_header=True,
                header_style="bold blue",
                box=box.ROUNDED
            )
            
            web_table.add_column("Source", style="cyan")
            web_table.add_column("URL", style="dim")
            web_table.add_column("Preview", style="dim")
            
            for i, web_result in enumerate(result["web_results"][:5], 1):
                title = web_result.get('title', 'Unknown')
                url = web_result.get('url', '')
                content = web_result.get('content', '')
                preview = content[:150] + "..." if len(content) > 150 else content
                
                web_table.add_row(
                    title,
                    url[:50] + "..." if len(url) > 50 else url,
                    preview
                )
            
            self.console.print(web_table)
            self.console.print()
        
        # Routing decision
        avg_score = result["average_score"]
        source = result["source"]
        
        # Color code based on source
        if source == "local":
            source_color = "[green]"
            border_style = "green"
        elif source == "hybrid":
            source_color = "[yellow]"
            border_style = "yellow"
        elif source == "web":
            source_color = "[blue]"
            border_style = "blue"
        else:
            source_color = "[red]"
            border_style = "red"
        
        # Build closing tag for source_color
        close_tag = source_color.replace('[', '[/')
        decision_panel = Panel(
            f"[bold]Average Relevance Score:[/bold] {source_color}{avg_score:.2f}/5.0{close_tag}\n"
            f"[bold]Source Decision:[/bold] {source_color}{source.upper()}{close_tag}\n"
            f"[dim]{result['reasoning']}[/dim]",
            box=box.ROUNDED,
            border_style=border_style
        )
        self.console.print(decision_panel)
        self.console.print()
        
        # Answer
        if result["source"] == "decline":
            answer_panel = Panel(
                f"[bold red]DECLINED TO ANSWER[/bold red]\n\n{result['answer']}",
                box=box.ROUNDED,
                border_style="red"
            )
        else:
            answer_panel = Panel(
                f"[bold green]ANSWER:[/bold green]\n\n{result['answer']}",
                box=box.ROUNDED,
                border_style="green"
            )
        
        self.console.print(answer_panel)
        self.console.print()


def parse_args(argv: List[str]) -> Dict:
    """Simple argument parser"""
    args = {
        "collection": "chunk_experiment_small",
        "top_k": 3,
        "mode": "test",
        "question": None,
    }
    
    i = 0
    rest = []
    while i < len(argv):
        tok = argv[i]
        if tok in ("--collection", "-c"):
            try:
                args["collection"] = argv[i + 1]
            except Exception:
                pass
            i += 2
            continue
        if tok in ("--topk", "--top_k", "-k"):
            try:
                args["top_k"] = int(argv[i + 1])
            except Exception:
                args["top_k"] = 3
            i += 2
            continue
        if tok in ("--single", "-s"):
            args["mode"] = "single"
            i += 1
            continue
        if tok in ("--help", "-h"):
            args["mode"] = "help"
            i += 1
            continue
        rest.append(tok)
        i += 1
    
    if rest:
        args["question"] = " ".join(rest).strip()
    
    return args


def main():
    """Main function"""
    console = Console()
    args = parse_args(sys.argv[1:])
    
    if args["mode"] == "help":
        console.print(
            Panel(
                "[bold]CRAG System (Corrective RAG with Web Search)[/bold]\n\n"
                "Examples:\n"
                "- Run test suite (3 questions):\n"
                "  `python3 crag_system.py`\n\n"
                "- Single question:\n"
                "  `python3 crag_system.py --single \"What are the 3 types of RAG?\"`\n\n"
                "- Choose collection:\n"
                "  `python3 crag_system.py --collection chunk_experiment_medium`\n\n"
                "[dim]Requires GEMINI_API_KEY for relevance judging.\n"
                "Requires TAVILY_API_KEY for web search (get free key at tavily.com).[/dim]",
                box=box.ROUNDED,
                border_style="cyan",
            )
        )
        return
    
    # Initialize CRAG
    crag = CRAG(
        collection_name=args["collection"]
    )
    
    console.print()
    console.print(Panel(
        f"[bold]CRAG System Configuration[/bold]\n"
        f"Collection: {crag.collection_name}\n"
        f"Top-K: {args['top_k']}\n"
        f"Web Search: {'Enabled' if crag.tavily_client else 'Disabled'}",
        box=box.ROUNDED,
        border_style="cyan"
    ))
    
    # Single question mode
    if args["mode"] == "single" and args["question"]:
        result = crag.process_question(args["question"], top_k=args["top_k"])
        crag.display_results(result)
        return
    
    # Test suite mode
    test_questions = [
        {
            "question": "What are the 3 types of RAG?",
            "expected": "local",
            "category": "Good local match (should use local)"
        },
        {
            "question": "What RAG research happened in January 2026?",
            "expected": "web",
            "category": "Bad local match (should use web)"
        },
        {
            "question": "Compare RAG to traditional search",
            "expected": "hybrid",
            "category": "Hybrid (should combine local + web)"
        }
    ]
    
    console.print()
    console.print(Panel(
        "[bold]TEST SUITE[/bold]\n"
        "Testing CRAG with 3 routing scenarios:\n"
        "1. Good local match → Local only\n"
        "2. Bad local match → Web search\n"
        "3. Hybrid → Local + Web",
        box=box.ROUNDED,
        border_style="yellow"
    ))
    
    results = []
    for test in test_questions:
        result = crag.process_question(test["question"], top_k=args["top_k"])
        result["expected"] = test["expected"]
        result["category"] = test["category"]
        results.append(result)
        crag.display_results(result)
        console.print("=" * 80)
    
    # Summary
    console.print()
    summary_table = Table(
        title="Test Suite Summary",
        show_header=True,
        header_style="bold green",
        box=box.ROUNDED
    )
    
    summary_table.add_column("Question", style="cyan")
    summary_table.add_column("Category", style="yellow")
    summary_table.add_column("Avg Score", justify="right", style="green")
    summary_table.add_column("Source", style="magenta")
    summary_table.add_column("Expected", style="dim")
    summary_table.add_column("Match", style="bold")
    
    for result in results:
        match = "✓" if result["source"] == result["expected"] else "✗"
        match_color = "[green]" if match == "✓" else "[yellow]"
        
        summary_table.add_row(
            result["question"][:40] + "..." if len(result["question"]) > 40 else result["question"],
            result["category"],
            f"{result['average_score']:.2f}",
            result["source"].upper(),
            result["expected"].upper(),
            f"{match_color}{match}[/{match_color}]"
        )
    
    console.print(summary_table)
    console.print()


if __name__ == "__main__":
    main()
