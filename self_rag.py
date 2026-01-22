"""
Self-RAG System with Retrieval Grading
Implements Self-RAG where Gemini acts as a JUDGE to score chunk relevance before answering.

Flow:
1. Retrieve top-k chunks from ChromaDB
2. Use Gemini to score relevance (1-5 scale) for each chunk
3. Calculate average relevance score
4. If average >= threshold: Generate answer
5. If average < threshold: Decline to answer ("I don't have enough information")

This proves the system can say "I don't know" when retrieval quality is poor.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

# Early progress messages
print("⚖️  Self-RAG System (Retrieval Grading)")
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


class SelfRAG:
    """Self-RAG system with retrieval grading"""
    
    def __init__(
        self,
        collection_name: str = "chunk_experiment_small",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        relevance_threshold: float = 3.0,
    ):
        """
        Initialize Self-RAG system
        
        Args:
            collection_name: ChromaDB collection to use
            embedding_model_name: Embedding model for retrieval
            relevance_threshold: Minimum average relevance score to answer (1-5 scale)
        """
        self.console = Console()
        self.collection_name = collection_name
        self.relevance_threshold = relevance_threshold
        
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
    
    def generate_answer(self, question: str, chunks: List[str]) -> str:
        """Generate answer from relevant chunks"""
        if not self.gemini_client:
            # Template fallback
            context = "\n\n".join(chunks[:2])
            return f"Based on the retrieved information: {context[:300]}..."
        
        context_text = "\n\n".join(chunks)
        prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Context:
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
            context = "\n\n".join(chunks[:2])
            return f"Based on the retrieved information: {context[:300]}..."
    
    def process_question(
        self,
        question: str,
        top_k: int = 3,
        show_details: bool = True
    ) -> Dict:
        """
        Process a question through Self-RAG pipeline
        
        Returns:
            Dictionary with results including decision, scores, answer, etc.
        """
        if show_details:
            self.console.print()
            self.console.print(Panel(
                f"[bold]Question:[/bold] {question}",
                box=box.ROUNDED,
                border_style="cyan"
            ))
        
        # Step 1: Retrieve chunks
        chunks, similarities, ids = self.retrieve_chunks(question, top_k)
        
        if not chunks:
            return {
                "question": question,
                "decision": "decline",
                "reason": "No chunks retrieved",
                "chunks": [],
                "scores": [],
                "average_score": 0.0,
                "answer": "I don't have enough information to answer this question."
            }
        
        # Step 2: Judge relevance
        if show_details:
            self.console.print("[dim]Judging relevance of retrieved chunks...[/dim]")
        
        relevance_scores = self.judge_relevance(question, chunks)
        average_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        # Step 3: Make decision
        if average_score < self.relevance_threshold:
            decision = "decline"
            answer = "I don't have enough information to answer this question."
            reason = f"Average relevance score ({average_score:.2f}) below threshold ({self.relevance_threshold})"
        else:
            decision = "answer"
            if show_details:
                self.console.print("[dim]Generating answer...[/dim]")
            answer = self.generate_answer(question, chunks)
            reason = f"Average relevance score ({average_score:.2f}) meets threshold ({self.relevance_threshold})"
        
        return {
            "question": question,
            "decision": decision,
            "reason": reason,
            "chunks": chunks,
            "similarities": similarities,
            "ids": ids,
            "relevance_scores": relevance_scores,
            "average_score": average_score,
            "answer": answer,
            "threshold": self.relevance_threshold
        }
    
    def display_results(self, result: Dict):
        """Display Self-RAG results beautifully"""
        self.console.print()
        self.console.print(Panel(
            "[bold]SELF-RAG RESULTS[/bold]",
            box=box.DOUBLE,
            border_style="cyan"
        ))
        self.console.print()
        
        # Retrieved chunks with scores
        chunks_table = Table(
            title="Retrieved Chunks with Relevance Scores",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )
        
        chunks_table.add_column("Chunk #", style="cyan", justify="right")
        chunks_table.add_column("Similarity", justify="right", style="yellow")
        chunks_table.add_column("Relevance Score", justify="right", style="green")
        chunks_table.add_column("Preview", style="dim")
        
        for i, (chunk, similarity, score) in enumerate(zip(
            result["chunks"],
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
        
        # Average score and decision
        avg_score = result["average_score"]
        threshold = result["threshold"]
        
        # Color code average
        if avg_score >= 4.0:
            avg_color = "[green]"
            decision_style = "green"
        elif avg_score >= 3.0:
            avg_color = "[yellow]"
            decision_style = "yellow"
        else:
            avg_color = "[red]"
            decision_style = "red"
        
        decision_panel = Panel(
            f"[bold]Average Relevance Score:[/bold] {avg_color}{avg_score:.2f}/5.0[/{avg_color}]\n"
            f"[bold]Threshold:[/bold] {threshold}\n"
            f"[bold]Decision:[/bold] [{decision_style}]{result['decision'].upper()}[/{decision_style}]\n"
            f"[dim]{result['reason']}[/dim]",
            box=box.ROUNDED,
            border_style=decision_style
        )
        self.console.print(decision_panel)
        self.console.print()
        
        # Answer or decline message
        if result["decision"] == "answer":
            answer_panel = Panel(
                f"[bold green]ANSWER:[/bold green]\n\n{result['answer']}",
                box=box.ROUNDED,
                border_style="green"
            )
        else:
            answer_panel = Panel(
                f"[bold red]DECLINED TO ANSWER[/bold red]\n\n{result['answer']}",
                box=box.ROUNDED,
                border_style="red"
            )
        
        self.console.print(answer_panel)
        self.console.print()


def parse_args(argv: List[str]) -> Dict:
    """Simple argument parser"""
    args = {
        "collection": "chunk_experiment_small",
        "threshold": 3.0,
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
        if tok in ("--threshold", "-t"):
            try:
                args["threshold"] = float(argv[i + 1])
            except Exception:
                args["threshold"] = 3.0
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
                "[bold]Self-RAG System (Retrieval Grading)[/bold]\n\n"
                "Examples:\n"
                "- Run test suite (3 questions):\n"
                "  `python3 self_rag.py`\n\n"
                "- Single question:\n"
                "  `python3 self_rag.py --single \"What are the 3 types of RAG?\"`\n\n"
                "- Custom threshold:\n"
                "  `python3 self_rag.py --threshold 4.0`\n\n"
                "- Choose collection:\n"
                "  `python3 self_rag.py --collection chunk_experiment_medium`\n\n"
                "[dim]Requires GEMINI_API_KEY for best results. Uses template-based judging as fallback.[/dim]",
                box=box.ROUNDED,
                border_style="cyan",
            )
        )
        return
    
    # Initialize Self-RAG
    self_rag = SelfRAG(
        collection_name=args["collection"],
        relevance_threshold=args["threshold"]
    )
    
    console.print()
    console.print(Panel(
        f"[bold]Self-RAG System Configuration[/bold]\n"
        f"Collection: {self_rag.collection_name}\n"
        f"Relevance Threshold: {self_rag.relevance_threshold}\n"
        f"Top-K: {args['top_k']}",
        box=box.ROUNDED,
        border_style="cyan"
    ))
    
    # Single question mode
    if args["mode"] == "single" and args["question"]:
        result = self_rag.process_question(args["question"], top_k=args["top_k"])
        self_rag.display_results(result)
        return
    
    # Test suite mode
    test_questions = [
        {
            "question": "What are the 3 types of RAG?",
            "expected": "answer",
            "category": "Good match"
        },
        {
            "question": "How does blockchain improve RAG?",
            "expected": "decline",
            "category": "Partial match (should fail)"
        },
        {
            "question": "What's the weather today?",
            "expected": "decline",
            "category": "Bad match (should fail)"
        }
    ]
    
    console.print()
    console.print(Panel(
        "[bold]TEST SUITE[/bold]\n"
        "Testing Self-RAG with 3 question types:\n"
        "1. Good match (should answer)\n"
        "2. Partial match (should decline)\n"
        "3. Bad match (should decline)",
        box=box.ROUNDED,
        border_style="yellow"
    ))
    
    results = []
    for test in test_questions:
        result = self_rag.process_question(test["question"], top_k=args["top_k"])
        result["expected"] = test["expected"]
        result["category"] = test["category"]
        results.append(result)
        self_rag.display_results(result)
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
    summary_table.add_column("Decision", style="magenta")
    summary_table.add_column("Expected", style="dim")
    summary_table.add_column("Match", style="bold")
    
    for result in results:
        match = "✓" if result["decision"] == result["expected"] else "✗"
        match_color = "[green]" if match == "✓" else "[red]"
        
        summary_table.add_row(
            result["question"][:40] + "..." if len(result["question"]) > 40 else result["question"],
            result["category"],
            f"{result['average_score']:.2f}",
            result["decision"].upper(),
            result["expected"].upper(),
            f"{match_color}{match}[/{match_color}]"
        )
    
    console.print(summary_table)
    console.print()


if __name__ == "__main__":
    main()
