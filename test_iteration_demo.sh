#!/bin/bash
# Demo script to test Agentic RAG iterative refinement
# This uses intentionally vague questions and higher thresholds to force iteration

echo "🤖 Agentic RAG Iteration Demo"
echo "=============================="
echo ""
echo "This demo uses:"
echo "- Intentionally vague questions"
echo "- Higher quality threshold (4.0-4.5)"
echo "- To force the system to iterate and refine"
echo ""
echo "Press Enter to start..."
read

python3 agentic_rag.py --demo
