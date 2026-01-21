"""
HyDE RAG Experiment (Day 8+)
Implements HyDE (Hypothetical Document Embeddings) on top of your Day 3 ChromaDB collections.

HyDE flow:
1) Take a user question
2) Use Gemini to generate a hypothetical answer (~200 words)
3) Embed the hypothetical answer (not the question)
4) Retrieve from ChromaDB using that embedding
5) Compare to standard retrieval using the question embedding

Outputs:
- The generated hypothetical answer
- Top-k chunks retrieved by HyDE vs standard
- A simple relevance score + "golden chunk" indicator
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

# Early progress messages (imports can be slow on first run)
print("🧪 HyDE RAG Experiment")
print("Loading dependencies (first run can take 10–60s)...")
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
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

sys.stdout.flush()


DEFAULT_COLLECTION_CANDIDATES = [
    "chunk_experiment_small",
    "chunk_experiment_medium",
    "chunk_experiment_large",
    "rag_documents",
]


@dataclass
class RetrievalResult:
    chunk_id: str
    similarity: float  # similarity from Chroma (1 - distance)
    doc: str
    relevance_score: float  # computed locally vs the original question


class HyDERAG:
    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chroma_path: str = "./chroma_db",
    ):
        self.console = Console()

        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )

        with self.console.status("[bold green]Loading embedding model...") as _:
            self.embedder = SentenceTransformer(embedding_model_name)

        self.collection_name = self._resolve_collection_name(collection_name)
        self.collection = self.client.get_collection(name=self.collection_name)

        self.gemini_model, self.gemini_rpm = self._init_gemini()

    def _resolve_collection_name(self, requested: Optional[str]) -> str:
        """Pick a collection to use; default to one of known Day 3 collections."""
        available = [c.name for c in self.client.list_collections()]

        if requested:
            if requested in available:
                return requested
            raise ValueError(
                f"Collection '{requested}' not found. Available: {', '.join(available) if available else '(none)'}"
            )

        for candidate in DEFAULT_COLLECTION_CANDIDATES:
            if candidate in available:
                return candidate

        if available:
            return available[0]

        raise ValueError("No ChromaDB collections found. Run `chunk_experiment.py` (or `rag_system.py`) first.")

    def _init_gemini(self) -> Tuple[Optional[object], Optional[int]]:
        """Initialize Gemini model for HyDE generation. Returns (model, rpm)."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.console.print(
                "[yellow]GEMINI_API_KEY not found. HyDE will fall back to a template hypothetical answer.[/yellow]"
            )
            return None, None

        if genai is None:
            self.console.print(
                "[yellow]google-generativeai not available. Install deps or re-run `pip3 install -r requirements.txt`.[/yellow]"
            )
            return None, None

        genai.configure(api_key=api_key)

        # Prefer higher RPM models first (same approach as your other scripts)
        models_to_try = [
            ("models/gemini-2.5-flash-lite", 10),
            ("models/gemma-3-12b-it", 30),
            ("models/gemma-3-4b-it", 30),
            ("models/gemini-2.5-flash", 5),
        ]
        for name, rpm in models_to_try:
            try:
                model = genai.GenerativeModel(name)
                self.console.print(f"[green]✓[/green] Gemini ready for HyDE: [bold]{name}[/bold] ({rpm} RPM)")
                return model, rpm
            except Exception:
                continue

        self.console.print("[yellow]Could not initialize any Gemini model; using template HyDE generation.[/yellow]")
        return None, None

    def _gemini_sleep(self):
        """Best-effort pacing for free-tier RPM limits."""
        if not self.gemini_rpm:
            return
        # Add a buffer; free tier can be spiky
        delay = max(6.0, (60.0 / float(self.gemini_rpm)) + 1.0)
        time.sleep(delay)

    def generate_hypothetical_answer(self, question: str, target_words: int = 200) -> str:
        """Generate a ~200-word hypothetical answer (HyDE)."""
        if not self.gemini_model:
            return self._template_hypo_answer(question, target_words=target_words)

        prompt = f"""You are generating a hypothetical answer ONLY to improve retrieval (HyDE).
Write an answer that sounds plausible and information-dense, but do NOT cite sources.
Length: ~{target_words} words.
Focus: include key terminology, synonyms, and subtopics that relevant documents would contain.

Question:
{question}

Hypothetical answer (~{target_words} words):"""

        try:
            # Pace calls a bit (especially on free tier)
            self._gemini_sleep()
            resp = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.4,
                    "max_output_tokens": 350,
                },
            )
            text = (resp.text or "").strip()
            return text if text else self._template_hypo_answer(question, target_words=target_words)
        except Exception as e:
            msg = str(e)
            self.console.print(f"[yellow]Gemini HyDE generation failed: {msg[:180]}... Using template fallback.[/yellow]")
            return self._template_hypo_answer(question, target_words=target_words)

    def _template_hypo_answer(self, question: str, target_words: int = 200) -> str:
        """Fallback hypothetical answer if Gemini is unavailable."""
        return (
            f"This is a hypothetical answer to improve retrieval for the question: {question}\n\n"
            "In Advanced RAG, query rewriting improves retrieval by translating the user’s intent into a form that "
            "matches the embedding space and the way documents were chunked. Common techniques include rewriting the "
            "query with clearer entities, adding missing context, expanding synonyms, decomposing complex questions "
            "into sub-queries, and generating alternative phrasings. This helps recall by surfacing relevant chunks "
            "that would be missed by a short or ambiguous query. It can also improve precision by removing noisy terms "
            "and focusing on discriminative keywords. Query rewriting is often paired with multi-query retrieval, HyDE, "
            "re-ranking, and routing. The main trade-offs are added latency and cost, plus the risk of semantic drift "
            "if the rewrite changes the question meaning. Guardrails include keeping the rewrite grounded, using multiple "
            "candidates, and evaluating with metrics like faithfulness and answer relevancy."
        )

    def _encode(self, text: str) -> List[float]:
        return self.embedder.encode([text]).tolist()[0]

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        # Avoid heavy numpy dependency: simple cosine
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / ((na**0.5) * (nb**0.5))

    def retrieve(self, query_embedding: List[float], top_k: int = 3) -> Tuple[List[str], List[float], List[str]]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        docs = results["documents"][0] if results.get("documents") else []
        distances = results["distances"][0] if results.get("distances") else []
        ids = results["ids"][0] if results.get("ids") else []
        sims = [1.0 - d for d in distances]
        return docs, sims, ids

    def score_relevance(self, question: str, docs: List[str]) -> List[float]:
        """Compute local relevance scores vs the original question (cosine sim in embedding space)."""
        q_emb = self._encode(question)
        doc_embs = self.embedder.encode(docs).tolist() if docs else []
        return [self._cosine_sim(q_emb, d_emb) for d_emb in doc_embs]

    def run_single_question(self, question: str, top_k: int = 3) -> Dict[str, object]:
        """Run standard vs HyDE retrieval for one question."""
        self.console.print(Panel(f"[bold]Question[/bold]\n{question}", box=box.ROUNDED, border_style="cyan"))

        with self.console.status("[bold green]Generating hypothetical answer (HyDE)...") as _:
            hypo = self.generate_hypothetical_answer(question, target_words=200)

        self.console.print(Panel(f"[bold]HyDE Hypothetical Answer (~200 words)[/bold]\n{hypo}", box=box.ROUNDED))

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=self.console) as p:
            t1 = p.add_task("Standard retrieval (embed question)...", total=None)
            q_emb = self._encode(question)
            std_docs, std_sims, std_ids = self.retrieve(q_emb, top_k=top_k)
            p.update(t1, completed=True)

            t2 = p.add_task("HyDE retrieval (embed hypothetical answer)...", total=None)
            h_emb = self._encode(hypo)
            hyde_docs, hyde_sims, hyde_ids = self.retrieve(h_emb, top_k=top_k)
            p.update(t2, completed=True)

        # Local relevance scoring (vs original question)
        std_rel = self.score_relevance(question, std_docs)
        hyde_rel = self.score_relevance(question, hyde_docs)

        std_results = [
            RetrievalResult(chunk_id=i, similarity=s, doc=d, relevance_score=r)
            for i, s, d, r in zip(std_ids, std_sims, std_docs, std_rel)
        ]
        hyde_results = [
            RetrievalResult(chunk_id=i, similarity=s, doc=d, relevance_score=r)
            for i, s, d, r in zip(hyde_ids, hyde_sims, hyde_docs, hyde_rel)
        ]

        # Golden chunk = highest relevance score among union
        union: Dict[str, RetrievalResult] = {}
        for rr in std_results + hyde_results:
            prev = union.get(rr.chunk_id)
            if (prev is None) or (rr.relevance_score > prev.relevance_score):
                union[rr.chunk_id] = rr
        golden = max(union.values(), key=lambda x: x.relevance_score) if union else None

        # Decide who found "more relevant context"
        std_avg = sum(std_rel) / len(std_rel) if std_rel else 0.0
        hyde_avg = sum(hyde_rel) / len(hyde_rel) if hyde_rel else 0.0
        winner = "HyDE" if hyde_avg > std_avg else "Standard" if std_avg > hyde_avg else "Tie"

        self._print_comparison_tables(std_results, hyde_results, golden, winner, std_avg, hyde_avg)

        return {
            "question": question,
            "hypothetical_answer": hypo,
            "standard": std_results,
            "hyde": hyde_results,
            "golden_chunk_id": golden.chunk_id if golden else None,
            "winner": winner,
            "avg_relevance_standard": std_avg,
            "avg_relevance_hyde": hyde_avg,
        }

    def _print_comparison_tables(
        self,
        standard: List[RetrievalResult],
        hyde: List[RetrievalResult],
        golden: Optional[RetrievalResult],
        winner: str,
        std_avg: float,
        hyde_avg: float,
    ):
        # Summary
        golden_id = golden.chunk_id if golden else "N/A"
        summary = Table(title="A/B Summary", show_header=True, header_style="bold magenta", box=box.ROUNDED)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Standard", justify="right", style="yellow")
        summary.add_column("HyDE", justify="right", style="green")
        summary.add_row("Avg relevance (vs question)", f"{std_avg:.4f}", f"{hyde_avg:.4f}")
        summary.add_row("Golden chunk id", golden_id[:32] + ("..." if len(golden_id) > 32 else ""), golden_id[:32] + ("..." if len(golden_id) > 32 else ""))
        summary.add_row("Winner (avg relevance)", winner, winner)
        self.console.print(summary)
        self.console.print()

        def mk_table(title: str, rows: List[RetrievalResult], border: str) -> Table:
            t = Table(title=title, show_header=True, header_style="bold", box=box.ROUNDED, border_style=border)
            t.add_column("#", justify="right", style="cyan", no_wrap=True)
            t.add_column("chunk_id", style="dim", no_wrap=True)
            t.add_column("Chroma sim", justify="right")
            t.add_column("Relevance", justify="right")
            t.add_column("Preview", overflow="fold")
            for idx, rr in enumerate(rows, 1):
                is_golden = golden is not None and rr.chunk_id == golden.chunk_id
                badge = "⭐ " if is_golden else ""
                preview = rr.doc.replace("\n", " ").strip()
                if len(preview) > 220:
                    preview = preview[:220] + "..."
                t.add_row(
                    f"{badge}{idx}",
                    rr.chunk_id[:28] + ("..." if len(rr.chunk_id) > 28 else ""),
                    f"{rr.similarity:.4f}",
                    f"{rr.relevance_score:.4f}",
                    preview,
                )
            return t

        self.console.print(mk_table("Standard Retrieval (embed question) - Top K", standard, border="yellow"))
        self.console.print()
        self.console.print(mk_table("HyDE Retrieval (embed hypothetical answer) - Top K", hyde, border="green"))
        self.console.print()

        # Golden chunk presence
        if golden:
            std_has = any(r.chunk_id == golden.chunk_id for r in standard)
            hyde_has = any(r.chunk_id == golden.chunk_id for r in hyde)
            self.console.print(
                Panel(
                    f"[bold]Golden chunk[/bold]: {golden.chunk_id}\n\n"
                    f"- Found by Standard: {'YES' if std_has else 'NO'}\n"
                    f"- Found by HyDE: {'YES' if hyde_has else 'NO'}\n\n"
                    f"[dim]Golden chunk = highest local relevance score (cosine similarity vs the original question).[/dim]",
                    box=box.ROUNDED,
                    border_style="cyan",
                )
            )
            self.console.print()


def parse_args(argv: List[str]) -> Dict[str, object]:
    args: Dict[str, object] = {
        "collection": None,
        "top_k": 3,
        "mode": "ab",  # "single" or "ab"
    }

    i = 0
    rest: List[str] = []
    while i < len(argv):
        tok = argv[i]
        if tok in ("--collection", "-c"):
            args["collection"] = argv[i + 1] if i + 1 < len(argv) else None
            i += 2
            continue
        if tok in ("--topk", "--top_k", "-k"):
            try:
                args["top_k"] = int(argv[i + 1])
            except Exception:
                args["top_k"] = 3
            i += 2
            continue
        if tok in ("--single",):
            args["mode"] = "single"
            i += 1
            continue
        if tok in ("--help", "-h"):
            args["mode"] = "help"
            i += 1
            continue
        rest.append(tok)
        i += 1

    args["question_override"] = " ".join(rest).strip() if rest else None
    return args


def main():
    console = Console()
    args = parse_args(sys.argv[1:])

    if args["mode"] == "help":
        console.print(
            Panel(
                "[bold]HyDE RAG Experiment[/bold]\n\n"
                "Examples:\n"
                "- Run A/B test (3 default questions):\n"
                "  `python3 hyde_rag.py`\n\n"
                "- Run single question:\n"
                "  `python3 hyde_rag.py --single \"Explain the benefits of query rewriting in Advanced RAG\"`\n\n"
                "- Choose collection + top-k:\n"
                "  `python3 hyde_rag.py --collection chunk_experiment_small --topk 3`\n\n"
                "[dim]Requires `chunk_experiment.py` collections for best results. HyDE generation uses GEMINI_API_KEY if set.[/dim]",
                box=box.ROUNDED,
                border_style="cyan",
            )
        )
        return

    hyde = HyDERAG(collection_name=args["collection"])

    console.print(
        Panel(
            f"[bold]Using ChromaDB collection:[/bold] `{hyde.collection_name}`\n"
            f"[bold]Embedding model:[/bold] `{hyde.embedder.__class__.__name__}`\n"
            f"[bold]Top K:[/bold] {args['top_k']}\n\n"
            "[dim]Tip: Use `--collection chunk_experiment_medium` to switch strategies.[/dim]",
            box=box.ROUNDED,
            border_style="green",
        )
    )
    console.print()

    if args["mode"] == "single":
        q = args["question_override"] or "Explain the benefits of query rewriting in Advanced RAG"
        hyde.run_single_question(q, top_k=int(args["top_k"]))
        return

    # A/B test set (as requested)
    questions = [
        "Explain the benefits of query rewriting",
        "Why does HyDE improve retrieval?",
        "What's the difference between HyDE and standard retrieval?",
    ]
    if args["question_override"]:
        # If user passes a question, run that plus the default set
        questions = [args["question_override"]] + questions

    console.print(Panel("[bold]A/B Test Set[/bold]\n" + "\n".join(f"- {q}" for q in questions), box=box.ROUNDED))
    console.print()

    results = []
    for q in questions:
        results.append(hyde.run_single_question(q, top_k=int(args["top_k"])))

    # Cost analysis (rough, narrative)
    console.print(Panel(
        "[bold]Cost / Latency Analysis (Back-of-the-napkin)[/bold]\n\n"
        "- Standard: 1 embedding call\n"
        "- HyDE: 1 LLM call (generate hypothetical answer) + 1 embedding call\n\n"
        "[bold]Trade-off[/bold]\n"
        "- HyDE can increase recall / surface different chunks\n"
        "- But adds latency + per-query API cost\n\n"
        "[dim]Rule of thumb: if HyDE consistently finds the golden chunk for your hardest queries, it's worth it.\n"
        "If it mostly adds redundancy/noise, stick to standard retrieval or use cheaper query rewriting/multi-query.[/dim]",
        box=box.ROUNDED,
        border_style="yellow"
    ))


if __name__ == "__main__":
    main()

