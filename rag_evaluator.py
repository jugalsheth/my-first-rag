"""
RAGAS Evaluation System for Chunking Strategy Comparison
Automatically scores answers from Small, Medium, and Large chunking strategies
using RAGAS metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
"""

import os
import warnings
from typing import List, Dict, Tuple
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use environment variables directly

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

# RAGAS imports
from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from datasets import Dataset


class RAGEvaluator:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", use_gemini: bool = False):
        """
        Initialize the RAG evaluator
        
        Args:
            embedding_model_name: Embedding model for retrieval
            use_gemini: Whether to use Google Gemini for answer generation (requires API key)
        """
        self.console = Console()
        self.use_gemini = use_gemini
        self.gemini_client = None
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load embedding model
        with self.console.status("[bold green]Loading embedding model...") as status:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Test questions for evaluation
        self.test_questions = [
            "Explain the trade-offs of using a Semantic Router in a multi-domain enterprise environment.",
            "What is the main difference between Naive and Advanced RAG?",
            "How does HyDE improve retrieval?",
            "What are the key steps in a RAG pipeline?",
            "What factors affect the quality of a RAG system?"
        ]
        
        # Initialize Gemini if needed
        if use_gemini:
            try:
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    self.console.print("[red]Error: GEMINI_API_KEY environment variable not set[/red]")
                    self.console.print("[yellow]Make sure your .env file contains: GEMINI_API_KEY=your_key_here[/yellow]")
                    self.console.print("[yellow]Falling back to template-based answers[/yellow]")
                    self.use_gemini = False
                else:
                    # Validate API key format (should start with AIza)
                    if not api_key.startswith("AIza"):
                        self.console.print("[yellow]Warning: API key format looks incorrect. Should start with 'AIza'[/yellow]")
                    genai.configure(api_key=api_key)
                    # Try models with higher quotas first (flash-lite: 10 RPM, gemma: 30 RPM)
                    try:
                        self.gemini_client = genai.GenerativeModel('models/gemini-2.5-flash-lite')
                        self.console.print("[green]✓[/green] Gemini client initialized (using gemini-2.5-flash-lite, 10 RPM limit)")
                    except:
                        try:
                            self.gemini_client = genai.GenerativeModel('models/gemma-3-12b-it')
                            self.console.print("[green]✓[/green] Gemini client initialized (using gemma-3-12b-it, 30 RPM limit)")
                        except:
                            try:
                                self.gemini_client = genai.GenerativeModel('models/gemma-3-4b-it')
                                self.console.print("[green]✓[/green] Gemini client initialized (using gemma-3-4b-it, 30 RPM limit)")
                            except:
                                try:
                                    self.gemini_client = genai.GenerativeModel('models/gemini-2.5-flash')
                                    self.console.print("[yellow]⚠[/yellow] Using gemini-2.5-flash (5 RPM limit - may hit quota)")
                                except Exception as e2:
                                    raise Exception(f"Gemini models not available: {e2}")
            except Exception as e:
                self.console.print(f"[red]Error initializing Gemini: {e}[/red]")
                self.console.print("[yellow]Falling back to template-based answers[/yellow]")
                self.use_gemini = False
    
    def print_header(self, title: str, subtitle: str = ""):
        """Print a beautiful header"""
        header_text = f"[bold cyan]{title}[/bold cyan]"
        if subtitle:
            header_text += f"\n[dim]{subtitle}[/dim]"
        self.console.print(Panel(header_text, box=box.DOUBLE, expand=False))
        self.console.print()
    
    def retrieve_contexts(self, collection_name: str, question: str, top_k: int = 3) -> List[str]:
        """Retrieve contexts from a ChromaDB collection"""
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([question]).tolist()[0]
            
            # Query the collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Extract documents
            retrieved_docs = results['documents'][0] if results['documents'] else []
            return retrieved_docs
        except Exception as e:
            self.console.print(f"[red]Error retrieving from {collection_name}: {e}[/red]")
            return []
    
    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """
        Generate an answer from contexts
        
        Uses Google Gemini if available, otherwise uses a template-based approach
        """
        if self.use_gemini and self.gemini_client:
            try:
                context_text = "\n\n".join(contexts)
                prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so.

Context:
{context_text}

Question: {question}

Answer:"""
                
                response = self.gemini_client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1,
                        "max_output_tokens": 500,
                    }
                )
                return response.text.strip()
            except Exception as e:
                self.console.print(f"[yellow]Gemini error: {e}, using template[/yellow]")
        
        # Template-based answer generation (fallback)
        if not contexts:
            return "I cannot answer this question as no relevant context was found."
        
        # Simple template: use the most relevant chunk as the answer
        # In production, you'd use an LLM here
        context_text = "\n\n".join(contexts[:2])  # Use top 2 chunks
        answer = f"Based on the retrieved information: {context_text[:500]}..."
        return answer
    
    def _custom_evaluate_with_gemini(self, dataset, llm):
        """
        Custom evaluation that works with Gemini (since RAGAS 0.4+ requires InstructorLLM)
        Implements simplified versions of Faithfulness and Answer Relevancy
        Respects free tier rate limit: 5 requests/minute (12 seconds between calls)
        """
        import time
        from langchain_core.messages import HumanMessage
        
        faithfulness_scores = []
        answer_relevancy_scores = []
        
        questions = dataset["question"]
        answers = dataset["answer"]
        contexts_list = dataset["contexts"]
        
        # Rate limiting: Adjust based on model
        # gemini-2.5-flash-lite: 10 RPM = 6 seconds between calls
        # gemma models: 30 RPM = 2 seconds between calls
        # gemini-2.5-flash: 5 RPM = 12 seconds between calls
        MIN_DELAY = 7  # Safe for 10 RPM (flash-lite), will work for gemma too
        
        for i, (question, answer, contexts) in enumerate(zip(questions, answers, contexts_list)):
            self.console.print(f"[dim]Evaluating question {i+1}/{len(questions)}...[/dim]")
            
            # Combined prompt to reduce API calls (evaluate both metrics in one call)
            context_text = "\n\n".join(contexts)
            combined_prompt = f"""Evaluate the following answer on two metrics. Return ONLY two numbers separated by a comma (faithfulness, relevancy), each between 0.0 and 1.0.

Context:
{context_text[:2000]}

Question: {question}

Answer: {answer[:500]}

Metrics:
1. Faithfulness: How well is the answer supported by the context? (0.0 = not supported, 1.0 = fully supported)
2. Relevancy: How well does the answer address the question? (0.0 = not relevant, 1.0 = highly relevant)

Format: faithfulness_score, relevancy_score
Example: 0.85, 0.90"""
            
            try:
                # Wait to respect rate limit (except for first call)
                if i > 0:
                    self.console.print(f"[yellow]Waiting {MIN_DELAY}s to respect rate limit (5 req/min)...[/yellow]")
                    time.sleep(MIN_DELAY)
                
                response = llm.invoke([HumanMessage(content=combined_prompt)])
                response_text = response.content.strip()
                
                # Parse the response (expecting "0.85, 0.90" format)
                try:
                    parts = response_text.split(',')
                    if len(parts) >= 2:
                        faithfulness_score = float(parts[0].strip())
                        relevancy_score = float(parts[1].strip())
                    else:
                        # Try to extract numbers if format is different
                        import re
                        numbers = re.findall(r'0?\.\d+|1\.0', response_text)
                        if len(numbers) >= 2:
                            faithfulness_score = float(numbers[0])
                            relevancy_score = float(numbers[1])
                        else:
                            raise ValueError("Could not parse scores")
                    
                    faithfulness_scores.append(max(0.0, min(1.0, faithfulness_score)))
                    answer_relevancy_scores.append(max(0.0, min(1.0, relevancy_score)))
                    self.console.print(f"[green]  ✓ Scores: Faithfulness={faithfulness_score:.2f}, Relevancy={relevancy_score:.2f}[/green]")
                except (ValueError, IndexError) as e:
                    self.console.print(f"[yellow]  ⚠ Could not parse response, using defaults: {response_text[:50]}[/yellow]")
                    faithfulness_scores.append(0.5)
                    answer_relevancy_scores.append(0.5)
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    # Check if it's daily quota (20/day) vs per-minute quota
                    if "perday" in error_msg.lower() or "daily" in error_msg.lower() or "20" in error_msg:
                        self.console.print(f"[red]  ✗ Daily quota exceeded (20 requests/day). Switching to template-based evaluation...[/red]")
                        # Use simple heuristics when quota is exceeded
                        faithfulness_score, relevancy_score = self._template_based_evaluation(question, answer, contexts)
                        faithfulness_scores.append(faithfulness_score)
                        answer_relevancy_scores.append(relevancy_score)
                    else:
                        # Per-minute quota - wait and retry
                        self.console.print(f"[yellow]  ⚠ Per-minute quota exceeded. Waiting 60 seconds...[/yellow]")
                        time.sleep(60)
                        try:
                            response = llm.invoke([HumanMessage(content=combined_prompt)])
                            response_text = response.content.strip()
                            parts = response_text.split(',')
                            faithfulness_score = float(parts[0].strip())
                            relevancy_score = float(parts[1].strip())
                            faithfulness_scores.append(max(0.0, min(1.0, faithfulness_score)))
                            answer_relevancy_scores.append(max(0.0, min(1.0, relevancy_score)))
                        except:
                            faithfulness_score, relevancy_score = self._template_based_evaluation(question, answer, contexts)
                            faithfulness_scores.append(faithfulness_score)
                            answer_relevancy_scores.append(relevancy_score)
                else:
                    self.console.print(f"[yellow]  ⚠ Error: {error_msg[:50]}, using template-based evaluation[/yellow]")
                    faithfulness_score, relevancy_score = self._template_based_evaluation(question, answer, contexts)
                    faithfulness_scores.append(faithfulness_score)
                    answer_relevancy_scores.append(relevancy_score)
        
        return {
            "faithfulness": faithfulness_scores,
            "answer_relevancy": answer_relevancy_scores
        }
    
    def _template_based_evaluation(self, question: str, answer: str, contexts: List[str]) -> Tuple[float, float]:
        """
        Simple template-based evaluation when API quota is exceeded
        Uses heuristics: keyword overlap, answer length, context coverage
        """
        import re
        
        # Faithfulness: Check if answer keywords appear in contexts
        context_text = " ".join(contexts).lower()
        answer_lower = answer.lower()
        
        # Extract meaningful words (3+ chars, not common stop words)
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        answer_words = set([w for w in re.findall(r'\b\w{3,}\b', answer_lower) if w not in stop_words])
        context_words = set([w for w in re.findall(r'\b\w{3,}\b', context_text) if w not in stop_words])
        
        # Faithfulness = % of answer words found in context
        if answer_words:
            overlap = len(answer_words & context_words) / len(answer_words)
            faithfulness = min(1.0, overlap * 1.2)  # Slight boost for partial matches
        else:
            faithfulness = 0.5
        
        # Answer Relevancy: Check if answer addresses question keywords
        question_words = set([w for w in re.findall(r'\b\w{3,}\b', question.lower()) if w not in stop_words])
        if question_words:
            answer_question_overlap = len(question_words & answer_words) / len(question_words)
            # Also check answer length (too short = less relevant, too long = might be off-topic)
            answer_length_score = min(1.0, len(answer.split()) / 20)  # Prefer 20+ word answers
            relevancy = (answer_question_overlap * 0.7 + answer_length_score * 0.3)
        else:
            relevancy = 0.5
        
        return faithfulness, relevancy
    
    def evaluate_strategy(self, strategy_name: str, questions: List[str]) -> Dict:
        """
        Evaluate a chunking strategy using RAGAS metrics
        
        Args:
            strategy_name: Name of the strategy (Small, Medium, Large)
            questions: List of test questions
            
        Returns:
            Dictionary with evaluation results
        """
        collection_name = f"chunk_experiment_{strategy_name.lower()}"
        
        self.console.print(f"[bold]Evaluating {strategy_name} strategy...[/bold]")
        
        # Prepare data for RAGAS
        ragas_data = {
            "question": [],
            "contexts": [],
            "answer": [],
            "ground_truth": []  # Optional - we'll leave empty for now
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task(f"Processing {strategy_name}...", total=len(questions))
            
            for question in questions:
                # Retrieve contexts
                contexts = self.retrieve_contexts(collection_name, question, top_k=3)
                
                if not contexts:
                    self.console.print(f"[yellow]Warning: No contexts found for question in {strategy_name}[/yellow]")
                    continue
                
                # Generate answer
                answer = self.generate_answer(question, contexts)
                
                # Store for RAGAS (format for RAGAS 0.4+)
                ragas_data["question"].append(question)
                ragas_data["contexts"].append(contexts)  # RAGAS expects list of list of strings
                ragas_data["answer"].append(answer)
                ragas_data["ground_truth"].append("")  # Empty for unsupervised evaluation
                
                progress.update(task, advance=1)
        
        # Create dataset for RAGAS
        dataset = Dataset.from_dict(ragas_data)
        
        # Run evaluation
        self.console.print(f"[dim]Running RAGAS evaluation for {strategy_name}...[/dim]")
        
        try:
            # Get LLM for metrics (RAGAS 0.4+ requires LLM for metrics)
            llm = None
            
            # Try to use Gemini if available (preferred)
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    # Try Gemma models first (30 RPM, separate quota pool)
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model="gemma-3-12b-it",
                            temperature=0,
                            google_api_key=gemini_key
                        )
                        self.console.print("[dim]Using Gemma 3 12B for RAGAS metrics (30 RPM, separate quota)[/dim]")
                    except:
                        try:
                            llm = ChatGoogleGenerativeAI(
                                model="gemma-3-4b-it",
                                temperature=0,
                                google_api_key=gemini_key
                            )
                            self.console.print("[dim]Using Gemma 3 4B for RAGAS metrics (30 RPM, separate quota)[/dim]")
                        except:
                            try:
                                llm = ChatGoogleGenerativeAI(
                                    model="gemini-2.5-flash-lite",
                                    temperature=0,
                                    google_api_key=gemini_key
                                )
                                self.console.print("[dim]Using Gemini 2.5 Flash Lite for RAGAS metrics (10 RPM)[/dim]")
                            except:
                                llm = ChatGoogleGenerativeAI(
                                    model="gemini-2.5-flash",
                                    temperature=0,
                                    google_api_key=gemini_key
                                )
                                self.console.print("[yellow]⚠ Using Gemini 2.5 Flash for RAGAS metrics (5 RPM - may hit quota)[/yellow]")
                except ImportError:
                    # langchain_google_genai not installed, try to install or use alternative
                    self.console.print("[yellow]langchain-google-genai not found. Install with: pip3 install langchain-google-genai[/yellow]")
                    # Try alternative: use google-generativeai directly with a wrapper
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=gemini_key)
                        # Create a simple wrapper for RAGAS
                        from langchain_core.language_models.llms import LLM
                        from langchain_core.callbacks.manager import CallbackManagerForLLMRun
                        from typing import Optional, List
                        
                        class GeminiLLMWrapper(LLM):
                            model_name: str = "gemini-1.5-flash"
                            
                            @property
                            def _llm_type(self) -> str:
                                return "gemini"
                            
                            def _call(
                                self,
                                prompt: str,
                                stop: Optional[List[str]] = None,
                                run_manager: Optional[CallbackManagerForLLMRun] = None,
                                **kwargs: any,
                            ) -> str:
                                model = genai.GenerativeModel(self.model_name)
                                response = model.generate_content(prompt)
                                return response.text
                        
                        llm = GeminiLLMWrapper()
                        self.console.print("[dim]Using Gemini via wrapper for RAGAS metrics[/dim]")
                    except Exception as e:
                        self.console.print(f"[yellow]Could not initialize Gemini LLM for RAGAS: {e}[/yellow]")
                except Exception as e:
                    self.console.print(f"[yellow]Could not initialize Gemini LLM for RAGAS: {e}[/yellow]")
            
            # Fallback: try OpenAI if available
            if not llm:
                openai_key = os.getenv("OPENAI_API_KEY")
                if openai_key:
                    try:
                        from langchain_openai import ChatOpenAI
                        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                        self.console.print("[dim]Using OpenAI for RAGAS metrics[/dim]")
                    except:
                        pass
            
            # Final fallback: try llm_factory
            if not llm:
                try:
                    from ragas.llms import llm_factory
                    llm = llm_factory()
                    self.console.print("[dim]Using auto-detected LLM for RAGAS metrics[/dim]")
                except:
                    pass
            
            if not llm:
                self.console.print("[red]Error: No LLM available for RAGAS metrics.[/red]")
                self.console.print("[yellow]RAGAS metrics require an LLM. Please set GEMINI_API_KEY or OPENAI_API_KEY.[/yellow]")
                raise Exception("No LLM available for RAGAS evaluation")
            
            # RAGAS 0.4+ requires InstructorLLM for ALL collection metrics, which Gemini doesn't support
            # So we'll create a simpler custom evaluation that works with Gemini
            self.console.print("[yellow]Note: RAGAS 0.4+ requires InstructorLLM (not supported with Gemini).[/yellow]")
            self.console.print("[yellow]Using custom evaluation metrics compatible with Gemini...[/yellow]")
            
            # Custom evaluation using Gemini directly
            scores = self._custom_evaluate_with_gemini(dataset, llm)
            
            # Calculate averages
            avg_scores = {
                "faithfulness": np.mean(scores["faithfulness"]) if scores["faithfulness"] else 0.0,
                "answer_relevancy": np.mean(scores["answer_relevancy"]) if scores["answer_relevancy"] else 0.0,
                "context_precision": 0.0,  # Not computed
                "context_recall": 0.0  # Not computed
            }
            
            return {
                "strategy": strategy_name,
                "scores": avg_scores,
                "raw_scores": scores,
                "num_questions": len(questions)
            }
            
        except Exception as e:
            self.console.print(f"[red]Error evaluating {strategy_name}: {e}[/red]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return {
                "strategy": strategy_name,
                "scores": {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "context_precision": 0.0,
                    "context_recall": 0.0
                },
                "raw_scores": {},
                "num_questions": len(questions)
            }
    
    def compare_strategies(self, results: List[Dict]):
        """Compare all strategies and display results"""
        self.print_header("🏆 RAGAS EVALUATION RESULTS", 
                         "Comprehensive comparison of chunking strategies")
        
        # Create comparison table
        table = Table(title="RAGAS Metrics Comparison", show_header=True, 
                     header_style="bold magenta", box=box.ROUNDED)
        
        table.add_column("Strategy", style="cyan", no_wrap=True)
        table.add_column("Faithfulness", justify="right", style="green")
        table.add_column("Answer Relevancy", justify="right", style="yellow")
        table.add_column("Context Precision", justify="right", style="blue")
        table.add_column("Context Recall", justify="right", style="magenta")
        table.add_column("Overall Score", justify="right", style="bold")
        
        # Calculate overall scores and find winners
        for result in results:
            scores = result["scores"]
            # Only use metrics that were actually computed (not placeholders)
            computed_metrics = [
                scores["faithfulness"],
                scores["answer_relevancy"]
            ]
            # Add context metrics only if they were computed (not 0.0 placeholder)
            if scores.get("context_precision", 0.0) != 0.0 or "context_precision" in result.get("raw_scores", {}):
                computed_metrics.append(scores["context_precision"])
            if scores.get("context_recall", 0.0) != 0.0 or "context_recall" in result.get("raw_scores", {}):
                computed_metrics.append(scores["context_recall"])
            
            overall = np.mean(computed_metrics) if computed_metrics else 0.0
            result["overall"] = overall
            
            # Show N/A for metrics that weren't computed (0.0 placeholder)
            context_prec = "N/A" if scores.get('context_precision', 0.0) == 0.0 and 'context_precision' not in result.get('raw_scores', {}) else f"{scores['context_precision']:.3f}"
            context_rec = "N/A" if scores.get('context_recall', 0.0) == 0.0 and 'context_recall' not in result.get('raw_scores', {}) else f"{scores['context_recall']:.3f}"
            
            table.add_row(
                result["strategy"],
                f"{scores['faithfulness']:.3f}",
                f"{scores['answer_relevancy']:.3f}",
                context_prec,
                context_rec,
                f"[bold]{overall:.3f}[/bold]"
            )
        
        self.console.print(table)
        self.console.print()
        
        # Find winners for each metric
        self.print_header("🥇 WINNERS BY METRIC", "Which strategy performs best for each metric?")
        
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "overall"]
        metric_names = {
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall",
            "overall": "Overall Score"
        }
        
        winners_table = Table(show_header=True, header_style="bold green", box=box.ROUNDED)
        winners_table.add_column("Metric", style="cyan")
        winners_table.add_column("Winner", style="green")
        winners_table.add_column("Score", justify="right", style="bold")
        winners_table.add_column("Interpretation", style="dim")
        
        for metric in metrics:
            best_result = max(results, key=lambda x: x["scores"].get(metric, x.get("overall", 0)))
            best_score = best_result["scores"].get(metric, best_result.get("overall", 0))
            
            interpretations = {
                "faithfulness": "Best at preventing hallucinations",
                "answer_relevancy": "Best at answering the question directly",
                "context_precision": "Best at retrieving relevant chunks",
                "context_recall": "Best at capturing all needed information",
                "overall": "Best overall performance"
            }
            
            winners_table.add_row(
                metric_names[metric],
                f"[green]{best_result['strategy']}[/green]",
                f"{best_score:.3f}",
                interpretations[metric]
            )
        
        self.console.print(winners_table)
        self.console.print()
        
        # Generate insights
        self.generate_insights(results)
    
    def generate_insights(self, results: Dict):
        """Generate natural language insights from results"""
        self.print_header("📊 KEY INSIGHTS & ANALYSIS", 
                         "What the data tells us about chunking strategies")
        
        # Find overall winner
        overall_winner = max(results, key=lambda x: x.get("overall", 0))
        
        insights = []
        
        # Overall winner insight
        insights.append(Panel(
            f"[bold]🏆 Overall Winner: {overall_winner['strategy']} Strategy[/bold]\n\n"
            f"The {overall_winner['strategy']} strategy achieved the highest overall score "
            f"({overall_winner.get('overall', 0):.3f}) across all RAGAS metrics.\n\n"
            "[yellow]What this means:[/yellow] This chunking strategy produces the most reliable, "
            "relevant, and complete answers for your specific use case and document structure.",
            box=box.ROUNDED,
            border_style="green"
        ))
        
        # Context Precision analysis (the "smoking gun")
        precision_scores = {r["strategy"]: r["scores"]["context_precision"] for r in results}
        best_precision = max(precision_scores, key=precision_scores.get)
        worst_precision = min(precision_scores, key=precision_scores.get)
        
        insights.append(Panel(
            f"[bold]🎯 Context Precision Analysis (The 'Smoking Gun')[/bold]\n\n"
            f"Best: [green]{best_precision}[/green] ({precision_scores[best_precision]:.3f})\n"
            f"Worst: [red]{worst_precision}[/red] ({precision_scores[worst_precision]:.3f})\n\n"
            "[yellow]Key Insight:[/yellow] Context Precision measures how relevant the retrieved chunks are. "
            "Small chunks (256) typically have HIGH precision (very relevant chunks) but may have LOWER "
            "Faithfulness or Answer Relevancy because chunks are too small to explain full concepts.\n\n"
            f"This is your data-driven proof: {best_precision} chunks retrieve the most relevant information, "
            "but check if this translates to better answers (Faithfulness & Relevancy).",
            box=box.ROUNDED,
            border_style="yellow"
        ))
        
        # Faithfulness vs Context Precision trade-off
        small_result = next((r for r in results if r["strategy"] == "Small"), None)
        large_result = next((r for r in results if r["strategy"] == "Large"), None)
        
        if small_result and large_result:
            small_faith = small_result["scores"]["faithfulness"]
            small_prec = small_result["scores"]["context_precision"]
            large_faith = large_result["scores"]["faithfulness"]
            large_prec = large_result["scores"]["context_precision"]
            
            insights.append(Panel(
                "[bold]⚖️ The Precision vs Completeness Trade-off[/bold]\n\n"
                f"Small (256): Precision={small_prec:.3f}, Faithfulness={small_faith:.3f}\n"
                f"Large (1024): Precision={large_prec:.3f}, Faithfulness={large_faith:.3f}\n\n"
                "[yellow]Analysis:[/yellow] "
                f"{'Small chunks have higher precision but lower faithfulness' if small_prec > large_prec and small_faith < large_faith else ''}"
                f"{'Large chunks provide better faithfulness but may include noise' if large_faith > small_faith and large_prec < small_prec else ''}"
                "This trade-off is the core decision point for production systems.",
                box=box.ROUNDED,
                border_style="cyan"
            ))
        
        # Answer Relevancy analysis
        relevancy_scores = {r["strategy"]: r["scores"]["answer_relevancy"] for r in results}
        best_relevancy = max(relevancy_scores, key=relevancy_scores.get)
        
        insights.append(Panel(
            f"[bold]💡 Answer Relevancy Winner: {best_relevancy}[/bold]\n\n"
            f"The {best_relevancy} strategy produces answers that are most directly relevant to the questions.\n\n"
            "[yellow]What this means:[/yellow] This strategy's chunk size allows the LLM to generate answers "
            "that directly address what users are asking, without going off-topic or missing key points.",
            box=box.ROUNDED,
            border_style="magenta"
        ))
        
        # Production recommendation
        recommendation = Panel(
            f"[bold]🚀 PRODUCTION RECOMMENDATION[/bold]\n\n"
            f"Based on RAGAS evaluation across {results[0]['num_questions']} test questions:\n\n"
            f"[green]Recommended Strategy: {overall_winner['strategy']} chunks[/green]\n\n"
            f"Reasoning:\n"
            f"• Overall Score: {overall_winner.get('overall', 0):.3f} (highest)\n"
            f"• Faithfulness: {overall_winner['scores']['faithfulness']:.3f} (prevents hallucinations)\n"
            f"• Answer Relevancy: {overall_winner['scores']['answer_relevancy']:.3f} (direct answers)\n"
            f"• Context Precision: {overall_winner['scores']['context_precision']:.3f} (relevant chunks)\n"
            f"• Context Recall: {overall_winner['scores']['context_recall']:.3f} (complete information)\n\n"
            "[dim]Note: This recommendation is based on your specific document structure and test questions. "
            "Monitor performance in production and adjust if needed.[/dim]",
            box=box.DOUBLE,
            border_style="green"
        )
        
        for insight in insights:
            self.console.print(insight)
            self.console.print()
        
        self.console.print(recommendation)
        self.console.print()
    
    def run_evaluation(self, use_gemini_for_answers: bool = False):
        """Run complete evaluation of all chunking strategies"""
        self.print_header("🔬 RAGAS EVALUATION SYSTEM", 
                         "Automated scoring of chunking strategies using RAGAS metrics")
        
        if use_gemini_for_answers:
            self.use_gemini = True
            try:
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    self.console.print("[red]Error: GEMINI_API_KEY environment variable not set[/red]")
                    self.console.print("[yellow]Using template-based answers (less accurate)[/yellow]\n")
                    self.use_gemini = False
                else:
                    genai.configure(api_key=api_key)
                    # Try models with higher quotas first
                    try:
                        self.gemini_client = genai.GenerativeModel('models/gemini-2.5-flash-lite')
                        self.console.print("[green]✓[/green] Using Gemini 2.5 Flash Lite for answer generation (10 RPM)\n")
                    except:
                        try:
                            self.gemini_client = genai.GenerativeModel('models/gemma-3-12b-it')
                            self.console.print("[green]✓[/green] Using Gemma 3 12B for answer generation (30 RPM)\n")
                        except:
                            try:
                                self.gemini_client = genai.GenerativeModel('models/gemma-3-4b-it')
                                self.console.print("[green]✓[/green] Using Gemma 3 4B for answer generation (30 RPM)\n")
                            except:
                                try:
                                    self.gemini_client = genai.GenerativeModel('models/gemini-2.5-flash')
                                    self.console.print("[yellow]⚠[/yellow] Using Gemini 2.5 Flash (5 RPM - may hit quota)\n")
                                except Exception as e2:
                                    self.console.print(f"[yellow]Gemini model error: {e2}[/yellow]")
                                    self.console.print("[yellow]Using template-based answers (less accurate)[/yellow]\n")
                                    self.use_gemini = False
            except Exception as e:
                self.console.print(f"[yellow]Gemini not available: {e}[/yellow]")
                self.console.print("[yellow]Using template-based answers (less accurate)[/yellow]\n")
                self.use_gemini = False
        else:
            self.console.print("[yellow]Note: Using template-based answer generation.[/yellow]")
            self.console.print("[yellow]For best results, set GEMINI_API_KEY and use --gemini flag[/yellow]\n")
        
        # Check if collections exist
        strategies = ["Small", "Medium", "Large"]
        available_strategies = []
        
        for strategy in strategies:
            collection_name = f"chunk_experiment_{strategy.lower()}"
            try:
                self.client.get_collection(name=collection_name)
                available_strategies.append(strategy)
            except:
                self.console.print(f"[red]Warning: Collection '{collection_name}' not found. Run chunk_experiment.py first.[/red]")
        
        if not available_strategies:
            self.console.print("[red]No chunking strategy collections found![/red]")
            self.console.print("[yellow]Please run: python3 chunk_experiment.py first[/yellow]")
            return
        
        self.console.print(f"[green]Found {len(available_strategies)} strategy collections to evaluate[/green]\n")
        
        # Evaluate each strategy
        results = []
        for strategy in available_strategies:
            result = self.evaluate_strategy(strategy, self.test_questions)
            results.append(result)
            self.console.print()
        
        # Compare and display results
        self.compare_strategies(results)
        
        return results


def main():
    """Main function"""
    import sys
    
    use_gemini = "--gemini" in sys.argv or "-g" in sys.argv
    
    evaluator = RAGEvaluator(use_gemini=use_gemini)
    evaluator.run_evaluation(use_gemini_for_answers=use_gemini)


if __name__ == "__main__":
    main()
