"""
Agentic RAG System with Iterative Refinement
The self-improving RAG that searches, evaluates, and refines until confident.

Flow:
1. Take user question
2. Attempt loop (max 3 iterations):
   - Retrieve chunks from ChromaDB
   - Generate answer
   - Self-grade answer quality (1-5 scale)
   - If score < 3.0: Refine query and try again
   - If score >= 3.0: Stop, return answer
3. Track all attempts (query, docs, answer, score)
4. Return best answer from all attempts

Key Innovation: The system improves its own retrieval through reflection and query refinement.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

# Early progress messages
print("🤖 Agentic RAG System (Iterative Refinement)")
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

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box


class AgenticRAG:
    """Agentic RAG system with iterative refinement"""
    
    def __init__(
        self,
        collection_name: str = "chunk_experiment_small",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        quality_threshold: float = 3.0,
        max_iterations: int = 3,
    ):
        """
        Initialize Agentic RAG system
        
        Args:
            collection_name: ChromaDB collection to use
            embedding_model_name: Embedding model for retrieval
            quality_threshold: Minimum answer quality score to accept (1-5 scale)
            max_iterations: Maximum number of refinement attempts
        """
        self.console = Console()
        self.collection_name = collection_name
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        
        # Rate limiting tracking
        self.last_api_call_time = 0
        self.request_count = 0
        self.min_delay = 7  # Minimum delay between calls (safer than 6)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load embedding model
        with self.console.status("[bold green]Loading embedding model...") as status:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize Gemini for answering, grading, and query refinement
        self.gemini_client = None
        self._init_gemini()
        
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
            self.console.print("[yellow]Warning: GEMINI_API_KEY not set. Will use template-based fallback.[/yellow]")
            return
        
        genai.configure(api_key=api_key)
        
        # Try models with higher quotas first (Gemma has 30 RPM vs 10 RPM for Flash Lite)
        models_to_try = [
            ('models/gemma-3-12b-it', 30, "Gemma 3 12B"),  # Highest limit first
            ('models/gemma-3-4b-it', 30, "Gemma 3 4B"),     # Second highest
            ('models/gemini-2.5-flash-lite', 10, "Gemini 2.5 Flash Lite"),  # Lower limit
            ('models/gemini-2.5-flash', 5, "Gemini 2.5 Flash")  # Lowest limit last
        ]
        
        for model_name, rpm, display_name in models_to_try:
            try:
                self.gemini_client = genai.GenerativeModel(model_name)
                self.rpm_limit = rpm
                self.model_name = model_name
                self.console.print(f"[green]✓[/green] Using {display_name} ({rpm} RPM)")
                self.console.print(f"[dim]Rate limit: {rpm} requests/minute = {60/rpm:.1f}s delay between calls[/dim]")
                return
            except Exception:
                continue
        
        self.console.print("[yellow]Warning: No Gemini models available. Using template-based fallback.[/yellow]")
    
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
    
    def _wait_for_rate_limit(self):
        """Wait to respect rate limits"""
        if not self.gemini_client:
            return
        
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        
        # Calculate required delay based on RPM limit
        required_delay = max(self.min_delay, 60.0 / self.rpm_limit)
        
        if time_since_last_call < required_delay:
            wait_time = required_delay - time_since_last_call
            if wait_time > 0.1:  # Only sleep if meaningful delay
                time.sleep(wait_time)
        
        self.last_api_call_time = time.time()
        self.request_count += 1
    
    def generate_answer(self, question: str, chunks: List[str]) -> str:
        """Generate answer from retrieved chunks"""
        if not self.gemini_client:
            # Template fallback - mark clearly as template
            if chunks:
                context = "\n\n".join(chunks[:2])
                # Make it clear this is a template (will be scored conservatively)
                return f"Based on the retrieved information: {context[:300]}..."
            return "I don't have enough information to answer this question."
        
        context_text = "\n\n".join(chunks) if chunks else "No relevant context found."
        
        prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context:
{context_text}

Question: {question}

Answer:"""
        
        try:
            # Rate limiting
            self._wait_for_rate_limit()
            
            response = self.gemini_client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 500,
                }
            )
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                self.console.print(f"[yellow]Rate limit hit, using template answer[/yellow]")
            else:
                self.console.print(f"[yellow]Error generating answer: {e}, using template[/yellow]")
            
            # Template fallback - mark clearly as template
            if chunks:
                context = "\n\n".join(chunks[:2])
                # Make it clear this is a template (will be scored conservatively)
                return f"Based on the retrieved information: {context[:300]}..."
            return "I don't have enough information to answer this question."
    
    def self_grade_answer(self, question: str, answer: str, chunks: List[str]) -> Tuple[float, str]:
        """
        Self-grade the answer quality (1-5 scale)
        
        Returns:
            (score, reasoning) tuple
        """
        if not self.gemini_client:
            # Template fallback: more conservative scoring
            if "don't have enough information" in answer.lower():
                return (1.0, "Answer indicates insufficient information")
            
            # Detect template answers (just concatenated chunks)
            is_template = (
                answer.lower().startswith("based on the retrieved information") or
                len(answer) < 100 or  # Very short answers are likely templates
                answer.count("...") > 0  # Truncated chunks
            )
            
            # Check if answer relates to chunks
            answer_words = set(answer.lower().split())
            chunk_words = set(" ".join(chunks).lower().split())
            overlap = len(answer_words & chunk_words) / len(answer_words) if answer_words else 0
            
            # More conservative scoring, especially for templates
            if is_template:
                # Template answers get capped at 2.5 max
                # High overlap just means chunks were retrieved, not that answer is good
                if overlap > 0.5:
                    score = 1.5 + (overlap - 0.5) * 2.0  # 1.5 to 2.5 range
                else:
                    score = 1.0 + overlap * 1.0  # 1.0 to 1.5 range
                return (min(2.5, max(1.0, score)), f"Template answer (conservative): overlap {overlap:.2f}")
            else:
                # Real answers (shouldn't happen without LLM, but just in case)
                if overlap > 0.3:
                    score = 2.0 + (overlap - 0.3) * 2.5  # 2.0 to 3.75 range (more conservative)
                else:
                    score = 1.0 + overlap * 1.5  # 1.0 to 2.0 range
                return (min(3.75, max(1.0, score)), f"Keyword overlap: {overlap:.2f}")
        
        prompt = f"""You are a quality judge for a RAG system. Evaluate how well this answer addresses the question.

Question: {question}

Answer:
{answer}

Retrieved Context (for reference):
{chr(10).join([f"- {chunk[:200]}..." for chunk in chunks[:3]])}

Rate the answer quality on a scale of 1-5:
- 1: Answer doesn't address the question or is completely wrong
- 2: Answer partially addresses question but missing key information
- 3: Answer adequately addresses question with some gaps
- 4: Answer well addresses question with minor gaps
- 5: Answer perfectly addresses question with all key information

Respond with ONLY a number (1-5) followed by a brief reason (one sentence).
Format: "SCORE: X. REASON: [your reason]"
"""
        
        try:
            # Rate limiting
            self._wait_for_rate_limit()
            
            response = self.gemini_client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 100,
                }
            )
            
            response_text = response.text.strip()
            
            # Parse score and reason
            import re
            score_match = re.search(r'SCORE:\s*(\d+)', response_text, re.IGNORECASE)
            reason_match = re.search(r'REASON:\s*(.+)', response_text, re.IGNORECASE)
            
            if score_match:
                score = float(score_match.group(1))
                score = max(1.0, min(5.0, score))
            else:
                # Fallback: extract first number
                numbers = re.findall(r'\d+', response_text)
                score = float(numbers[0]) if numbers else 3.0
                score = max(1.0, min(5.0, score))
            
            reason = reason_match.group(1).strip() if reason_match else "No reason provided"
            
            return (score, reason)
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                self.console.print(f"[yellow]Rate limit hit for grading, using conservative fallback[/yellow]")
            else:
                self.console.print(f"[yellow]Error grading answer: {e}, using conservative fallback[/yellow]")
            
            # Conservative fallback scoring (same logic as when Gemini unavailable)
            if "don't have enough information" in answer.lower():
                return (1.0, "Answer indicates insufficient information")
            
            # Detect template answers
            is_template = (
                answer.lower().startswith("based on the retrieved information") or
                len(answer) < 100 or
                answer.count("...") > 0
            )
            
            answer_words = set(answer.lower().split())
            chunk_words = set(" ".join(chunks).lower().split())
            overlap = len(answer_words & chunk_words) / len(answer_words) if answer_words else 0
            
            if is_template:
                # Template answers capped at 2.5 max
                if overlap > 0.5:
                    score = 1.5 + (overlap - 0.5) * 2.0
                else:
                    score = 1.0 + overlap * 1.0
                return (min(2.5, max(1.0, score)), f"Fallback (template, conservative): overlap {overlap:.2f}")
            else:
                # Real answers (more conservative than before)
                if overlap > 0.3:
                    score = 2.0 + (overlap - 0.3) * 2.5
                else:
                    score = 1.0 + overlap * 1.5
                return (min(3.75, max(1.0, score)), f"Fallback: overlap {overlap:.2f}")
    
    def refine_query(self, original_question: str, previous_query: str, answer: str, score: float, chunks: List[str]) -> str:
        """
        Refine the query for better retrieval
        
        Args:
            original_question: The original user question
            previous_query: The query used in previous attempt
            answer: The answer generated
            score: The quality score received
            chunks: The chunks retrieved
        
        Returns:
            Refined query string
        """
        if not self.gemini_client:
            # Template fallback: add more specific terms
            if score < 2.0:
                # Very low score, try to make more specific
                return f"{original_question} specific techniques methods"
            elif score < 3.0:
                # Low score, add context
                return f"{original_question} detailed explanation"
            return previous_query
        
        prompt = f"""You are a query refinement expert. The previous query didn't retrieve good enough information.

Original Question: {original_question}
Previous Query: {previous_query}
Previous Answer Quality Score: {score}/5.0

Previous Answer:
{answer[:300]}

The answer quality was too low. Refine the query to retrieve better information. Make it:
- More specific and targeted
- Include key terms that might be in relevant documents
- Focus on what information is actually needed

Respond with ONLY the refined query. No explanation, just the new query."""
        
        try:
            # Rate limiting
            self._wait_for_rate_limit()
            
            response = self.gemini_client.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.5,  # Higher temperature for creativity
                    "max_output_tokens": 100,
                }
            )
            
            refined = response.text.strip()
            # Clean up common prefixes
            refined = refined.replace("Refined query:", "").replace("Query:", "").strip()
            return refined if refined else previous_query
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                self.console.print(f"[yellow]Rate limit hit for refinement, using template[/yellow]")
            else:
                self.console.print(f"[yellow]Error refining query: {e}, using template[/yellow]")
            
            # Template fallback
            if score < 2.0:
                return f"{original_question} specific techniques methods"
            elif score < 3.0:
                return f"{original_question} detailed explanation"
            return previous_query
    
    def process_question(
        self,
        question: str,
        top_k: int = 3,
        show_details: bool = True
    ) -> Dict:
        """
        Process a question through Agentic RAG pipeline with iterative refinement
        
        Returns:
            Dictionary with all attempts, best answer, and final decision
        """
        if show_details:
            self.console.print()
            self.console.print(Panel(
                f"[bold]Question:[/bold] {question}",
                box=box.ROUNDED,
                border_style="cyan"
            ))
        
        attempts = []
        current_query = question
        best_attempt = None
        best_score = 0.0
        
        # Iterative refinement loop
        for iteration in range(1, self.max_iterations + 1):
            if show_details:
                self.console.print()
                self.console.print(Panel(
                    f"[bold]Attempt {iteration}/{self.max_iterations}[/bold]",
                    box=box.ROUNDED,
                    border_style="yellow"
                ))
                self.console.print(f"[dim]Query: {current_query}[/dim]")
            
            # Step 1: Retrieve chunks
            if show_details:
                self.console.print("[dim]Retrieving chunks...[/dim]")
            
            chunks, similarities, ids = self.retrieve_chunks(current_query, top_k)
            
            if not chunks:
                # No chunks found
                answer = "I don't have enough information to answer this question."
                score, reason = self.self_grade_answer(question, answer, [])
                
                attempt = {
                    "iteration": iteration,
                    "query": current_query,
                    "chunks": [],
                    "similarities": [],
                    "answer": answer,
                    "score": score,
                    "reason": reason,
                    "stopped": True,
                    "reason_stopped": "No chunks retrieved"
                }
                attempts.append(attempt)
                
                if score > best_score:
                    best_score = score
                    best_attempt = attempt
                
                break
            
            # Step 2: Generate answer
            if show_details:
                self.console.print("[dim]Generating answer...[/dim]")
            
            answer = self.generate_answer(question, chunks)
            
            # Step 3: Self-grade
            if show_details:
                self.console.print("[dim]Self-grading answer quality...[/dim]")
            
            score, reason = self.self_grade_answer(question, answer, chunks)
            
            attempt = {
                "iteration": iteration,
                "query": current_query,
                "chunks": chunks,
                "similarities": similarities,
                "answer": answer,
                "score": score,
                "reason": reason,
                "stopped": False,
                "reason_stopped": None
            }
            attempts.append(attempt)
            
            # Track best attempt
            if score > best_score:
                best_score = score
                best_attempt = attempt
            
            # Step 4: Check if good enough
            if score >= self.quality_threshold:
                attempt["stopped"] = True
                attempt["reason_stopped"] = f"Quality threshold met ({score:.2f} >= {self.quality_threshold})"
                if show_details:
                    self.console.print(f"[green]✓ Quality threshold met! Score: {score:.2f}/5.0[/green]")
                break
            
            # Step 5: Refine query for next iteration
            if iteration < self.max_iterations:
                if show_details:
                    self.console.print(f"[yellow]Quality below threshold ({score:.2f} < {self.quality_threshold}). Refining query...[/yellow]")
                
                current_query = self.refine_query(question, current_query, answer, score, chunks)
            else:
                attempt["stopped"] = True
                attempt["reason_stopped"] = f"Max iterations reached (best score: {best_score:.2f})"
                if show_details:
                    self.console.print(f"[yellow]Max iterations reached. Best score: {best_score:.2f}/5.0[/yellow]")
        
        return {
            "question": question,
            "attempts": attempts,
            "best_attempt": best_attempt,
            "best_score": best_score,
            "total_iterations": len(attempts),
            "quality_threshold": self.quality_threshold
        }
    
    def display_results(self, result: Dict):
        """Display Agentic RAG results beautifully"""
        self.console.print()
        self.console.print(Panel(
            "[bold]AGENTIC RAG RESULTS[/bold]",
            box=box.DOUBLE,
            border_style="cyan"
        ))
        self.console.print()
        
        # Attempts table
        attempts_table = Table(
            title="Iteration History",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )
        
        attempts_table.add_column("Attempt", style="cyan", justify="right")
        attempts_table.add_column("Query", style="yellow")
        attempts_table.add_column("Score", justify="right", style="green")
        attempts_table.add_column("Status", style="magenta")
        
        for attempt in result["attempts"]:
            # Color code score
            score = attempt["score"]
            if score >= 4.0:
                score_color = "[green]"
            elif score >= 3.0:
                score_color = "[yellow]"
            else:
                score_color = "[red]"
            
            status = "✓ Stopped" if attempt["stopped"] else "→ Continue"
            if attempt["stopped"]:
                status_color = "[green]"
            else:
                status_color = "[dim]"
            
            query_preview = attempt["query"][:50] + "..." if len(attempt["query"]) > 50 else attempt["query"]
            
            attempts_table.add_row(
                str(attempt["iteration"]),
                query_preview,
                f"{score_color}{score:.2f}/5.0[/{score_color}]",
                f"{status_color}{status}[/{status_color}]"
            )
        
        self.console.print(attempts_table)
        self.console.print()
        
        # Best attempt details
        best = result["best_attempt"]
        if best:
            best_panel = Panel(
                f"[bold]Best Attempt: #{best['iteration']}[/bold]\n"
                f"[bold]Query:[/bold] {best['query']}\n"
                f"[bold]Score:[/bold] [green]{best['score']:.2f}/5.0[/green]\n"
                f"[bold]Reason:[/bold] {best['reason']}\n"
                f"[bold]Stopped:[/bold] {best['reason_stopped'] or 'No'}",
                box=box.ROUNDED,
                border_style="green"
            )
            self.console.print(best_panel)
            self.console.print()
            
            # Answer
            answer_panel = Panel(
                f"[bold green]FINAL ANSWER:[/bold green]\n\n{best['answer']}",
                box=box.ROUNDED,
                border_style="green"
            )
            self.console.print(answer_panel)
            self.console.print()
            
            # Chunks from best attempt
            if best["chunks"]:
                chunks_table = Table(
                    title=f"Retrieved Chunks (Attempt #{best['iteration']})",
                    show_header=True,
                    header_style="bold blue",
                    box=box.ROUNDED
                )
                
                chunks_table.add_column("Chunk #", style="cyan", justify="right")
                chunks_table.add_column("Similarity", justify="right", style="yellow")
                chunks_table.add_column("Preview", style="dim")
                
                for i, (chunk, similarity) in enumerate(zip(best["chunks"], best["similarities"]), 1):
                    preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                    chunks_table.add_row(
                        str(i),
                        f"{similarity:.4f}",
                        preview
                    )
                
                self.console.print(chunks_table)
                self.console.print()


def parse_args(argv: List[str]) -> Dict:
    """Simple argument parser"""
    args = {
        "collection": "chunk_experiment_small",
        "top_k": 3,
        "threshold": 3.0,
        "max_iterations": 3,
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
        if tok in ("--threshold", "-t"):
            try:
                args["threshold"] = float(argv[i + 1])
            except Exception:
                args["threshold"] = 3.0
            i += 2
            continue
        if tok in ("--iterations", "-i"):
            try:
                args["max_iterations"] = int(argv[i + 1])
            except Exception:
                args["max_iterations"] = 3
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
                "[bold]Agentic RAG System (Iterative Refinement)[/bold]\n\n"
                "Examples:\n"
                "- Run test suite (3 questions):\n"
                "  `python3 agentic_rag.py`\n\n"
                "- Single question:\n"
                "  `python3 agentic_rag.py --single \"How can I improve RAG retrieval accuracy?\"`\n\n"
                "- Custom threshold:\n"
                "  `python3 agentic_rag.py --threshold 4.0`\n\n"
                "- More iterations:\n"
                "  `python3 agentic_rag.py --iterations 5`\n\n"
                "[dim]Requires GEMINI_API_KEY for best results. Uses template-based fallback if unavailable.[/dim]",
                box=box.ROUNDED,
                border_style="cyan",
            )
        )
        return
    
    # Initialize Agentic RAG
    agentic_rag = AgenticRAG(
        collection_name=args["collection"],
        quality_threshold=args["threshold"],
        max_iterations=args["max_iterations"]
    )
    
    console.print()
    console.print(Panel(
        f"[bold]Agentic RAG Configuration[/bold]\n"
        f"Collection: {agentic_rag.collection_name}\n"
        f"Top-K: {args['top_k']}\n"
        f"Quality Threshold: {agentic_rag.quality_threshold}\n"
        f"Max Iterations: {agentic_rag.max_iterations}",
        box=box.ROUNDED,
        border_style="cyan"
    ))
    
    # Single question mode
    if args["mode"] == "single" and args["question"]:
        result = agentic_rag.process_question(args["question"], top_k=args["top_k"])
        agentic_rag.display_results(result)
        return
    
    # Test suite mode
    test_questions = [
        {
            "question": "How do I make RAG better?",
            "expected_iterations": "2-3",
            "category": "Vague question (forces refinement)"
        },
        {
            "question": "What is the difference between Naive and Advanced RAG?",
            "expected_iterations": "1",
            "category": "Specific question (one shot)"
        },
        {
            "question": "What's the weather today?",
            "expected_iterations": "3",
            "category": "Impossible question (max iterations, all fail)"
        }
    ]
    
    console.print()
    console.print(Panel(
        "[bold]TEST SUITE[/bold]\n"
        "Testing Agentic RAG with 3 question types:\n"
        "1. Vague question (should refine 2-3 times)\n"
        "2. Specific question (should nail it first try)\n"
        "3. Impossible question (should try 3 times, then decline)",
        box=box.ROUNDED,
        border_style="yellow"
    ))
    
    results = []
    for i, test in enumerate(test_questions, 1):
        console.print(f"\n[bold yellow]Running test {i}/{len(test_questions)}...[/bold yellow]")
        result = agentic_rag.process_question(test["question"], top_k=args["top_k"])
        result["expected_iterations"] = test["expected_iterations"]
        result["category"] = test["category"]
        results.append(result)
        agentic_rag.display_results(result)
        console.print("=" * 80)
        
        # Add delay between questions to avoid rate limits
        if i < len(test_questions):
            delay = max(10, 60 / agentic_rag.rpm_limit * 2)  # 2x safety margin
            console.print(f"[dim]Waiting {delay:.1f}s before next question to respect rate limits...[/dim]")
            time.sleep(delay)
    
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
    summary_table.add_column("Iterations", justify="right", style="green")
    summary_table.add_column("Best Score", justify="right", style="magenta")
    summary_table.add_column("Expected", style="dim")
    summary_table.add_column("Match", style="bold")
    
    for result in results:
        iterations = result["total_iterations"]
        expected = result["expected_iterations"]
        
        # Check if iterations match expectation
        if "-" in expected:
            # Range like "2-3"
            min_exp, max_exp = map(int, expected.split("-"))
            match = "✓" if min_exp <= iterations <= max_exp else "✗"
        else:
            # Single number
            match = "✓" if iterations == int(expected) else "✗"
        
        match_color = "[green]" if match == "✓" else "[yellow]"
        
        summary_table.add_row(
            result["question"][:40] + "..." if len(result["question"]) > 40 else result["question"],
            result["category"],
            str(iterations),
            f"{result['best_score']:.2f}",
            expected,
            f"{match_color}{match}[/{match_color}]"
        )
    
    console.print(summary_table)
    console.print()


if __name__ == "__main__":
    main()
