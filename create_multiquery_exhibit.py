"""
Create Multi-Query RAG Research Exhibit PNG
Visual summary of the Multi-Query RAG experiment findings
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
from datetime import datetime

# Set up the figure
fig = plt.figure(figsize=(16, 10), facecolor='white')
ax = fig.add_subplot(111)
ax.axis('off')

# Colors
COLORS = {
    'header': '#1a1a2e',
    'accent': '#0f3460',
    'primary': '#16213e',
    'secondary': '#e94560',
    'success': '#4a90e2',
    'warning': '#f39c12',
    'info': '#3498db',
    'light': '#ecf0f1',
    'dark': '#2c3e50',
    'query1': '#e74c3c',
    'query2': '#3498db',
    'query3': '#2ecc71',
    'query4': '#f39c12'
}

# Title Section
title_y = 0.96
ax.text(0.5, title_y, 'MULTI-QUERY RAG EXPERIMENT', 
        fontsize=36, fontweight='bold', 
        ha='center', va='top', color=COLORS['header'])
ax.text(0.5, title_y - 0.035, 'Query Expansion for Improved Retrieval Coverage', 
        fontsize=20, style='italic',
        ha='center', va='top', color=COLORS['accent'])
ax.text(0.5, title_y - 0.06, f'Day 8 Research • {datetime.now().strftime("%B %d, %Y")}',
        fontsize=14, ha='center', va='top', color=COLORS['dark'])

# Methodology Section (Top Left)
method_x = 0.05
method_y = 0.85

method_box = FancyBboxPatch(
    (method_x, method_y - 0.25), 0.42, 0.25,
    boxstyle="round,pad=0.015", 
    facecolor=COLORS['light'], 
    edgecolor=COLORS['accent'],
    linewidth=2,
    zorder=2
)
ax.add_patch(method_box)

ax.text(method_x + 0.21, method_y, 'METHODOLOGY', 
        fontsize=16, fontweight='bold', 
        ha='center', va='top', color=COLORS['header'])

methodology_steps = [
    '1. Take original user question',
    '2. Use Gemini to generate 3 query variations',
    '3. Search ChromaDB with all 4 queries (original + 3)',
    '4. Combine and deduplicate results',
    '5. Compare to single-query baseline',
    '6. Calculate coverage improvement'
]

for i, step in enumerate(methodology_steps):
    y_pos = method_y - 0.04 - (i * 0.032)
    ax.text(method_x + 0.02, y_pos, step, 
            fontsize=11, ha='left', va='top', color=COLORS['dark'])

# Key Metrics Section (Top Right)
metrics_x = 0.53
metrics_y = 0.85

metrics_box = FancyBboxPatch(
    (metrics_x, metrics_y - 0.25), 0.42, 0.25,
    boxstyle="round,pad=0.015", 
    facecolor=COLORS['light'], 
    edgecolor=COLORS['success'],
    linewidth=2,
    zorder=2
)
ax.add_patch(metrics_box)

ax.text(metrics_x + 0.21, metrics_y, 'KEY METRICS', 
        fontsize=16, fontweight='bold', 
        ha='center', va='top', color=COLORS['header'])

metrics_list = [
    'Coverage Improvement: (Multi - Single) / Single × 100%',
    'New Chunks Found: Unique chunks only multi-query discovered',
    'Overlap Analysis: Which chunks both approaches found',
    'Token Cost: 3x queries = 3x API calls',
    'Query Diversity: How different are the variations?'
]

for i, metric in enumerate(metrics_list):
    y_pos = metrics_y - 0.04 - (i * 0.032)
    ax.text(metrics_x + 0.02, y_pos, metric, 
            fontsize=10, ha='left', va='top', color=COLORS['dark'])

# Query Flow Diagram (Center)
flow_y = 0.50

ax.text(0.5, flow_y + 0.08, 'QUERY EXPANSION FLOW', 
        fontsize=18, fontweight='bold', 
        ha='center', va='bottom', color=COLORS['header'])

# Original Query Box
orig_box = FancyBboxPatch(
    (0.15, flow_y - 0.05), 0.20, 0.10,
    boxstyle="round,pad=0.01", 
    facecolor=COLORS['query1'], 
    edgecolor=COLORS['accent'],
    linewidth=2,
    alpha=0.8,
    zorder=3
)
ax.add_patch(orig_box)

ax.text(0.25, flow_y, 'Original Query', 
        fontsize=12, fontweight='bold', 
        ha='center', va='center', color='white')

# Arrow to Gemini
arrow1 = FancyArrowPatch((0.35, flow_y), (0.45, flow_y),
                         arrowstyle='->', lw=3, 
                         color=COLORS['accent'], zorder=2)
ax.add_patch(arrow1)

# Gemini Box
gemini_box = FancyBboxPatch(
    (0.45, flow_y - 0.05), 0.10, 0.10,
    boxstyle="round,pad=0.01", 
    facecolor=COLORS['warning'], 
    edgecolor=COLORS['accent'],
    linewidth=2,
    alpha=0.8,
    zorder=3
)
ax.add_patch(gemini_box)

ax.text(0.50, flow_y, 'Gemini\nQuery\nExpansion', 
        fontsize=9, fontweight='bold', 
        ha='center', va='center', color='white')

# Arrow from Gemini
arrow2 = FancyArrowPatch((0.55, flow_y), (0.65, flow_y),
                         arrowstyle='->', lw=3, 
                         color=COLORS['accent'], zorder=2)
ax.add_patch(arrow2)

# Query Variations Box
var_box = FancyBboxPatch(
    (0.65, flow_y - 0.10), 0.20, 0.20,
    boxstyle="round,pad=0.01", 
    facecolor=COLORS['success'], 
    edgecolor=COLORS['accent'],
    linewidth=2,
    alpha=0.8,
    zorder=3
)
ax.add_patch(var_box)

ax.text(0.75, flow_y + 0.05, '3 Query', 
        fontsize=11, fontweight='bold', 
        ha='center', va='center', color='white')
ax.text(0.75, flow_y, 'Variations', 
        fontsize=11, fontweight='bold', 
        ha='center', va='center', color='white')

# Arrows down to retrieval
arrow3 = FancyArrowPatch((0.25, flow_y - 0.05), (0.25, flow_y - 0.20),
                         arrowstyle='->', lw=2, 
                         color=COLORS['accent'], zorder=2)
ax.add_patch(arrow3)

arrow4 = FancyArrowPatch((0.50, flow_y - 0.05), (0.50, flow_y - 0.20),
                         arrowstyle='->', lw=2, 
                         color=COLORS['accent'], zorder=2)
ax.add_patch(arrow4)

arrow5 = FancyArrowPatch((0.75, flow_y - 0.10), (0.75, flow_y - 0.20),
                         arrowstyle='->', lw=2, 
                         color=COLORS['accent'], zorder=2)
ax.add_patch(arrow5)

# ChromaDB Retrieval Box
retrieval_box = FancyBboxPatch(
    (0.10, flow_y - 0.35), 0.80, 0.12,
    boxstyle="round,pad=0.01", 
    facecolor=COLORS['info'], 
    edgecolor=COLORS['accent'],
    linewidth=2,
    alpha=0.7,
    zorder=3
)
ax.add_patch(retrieval_box)

ax.text(0.50, flow_y - 0.25, 'ChromaDB Retrieval (4 Queries)', 
        fontsize=13, fontweight='bold', 
        ha='center', va='center', color='white')
ax.text(0.50, flow_y - 0.30, 'Each query retrieves top-k chunks', 
        fontsize=10, 
        ha='center', va='center', color='white', style='italic')

# Arrow down to deduplication
arrow6 = FancyArrowPatch((0.50, flow_y - 0.35), (0.50, flow_y - 0.45),
                         arrowstyle='->', lw=3, 
                         color=COLORS['secondary'], zorder=2)
ax.add_patch(arrow6)

# Deduplication Box
dedup_box = FancyBboxPatch(
    (0.30, flow_y - 0.55), 0.40, 0.08,
    boxstyle="round,pad=0.01", 
    facecolor=COLORS['secondary'], 
    edgecolor=COLORS['accent'],
    linewidth=2,
    alpha=0.8,
    zorder=3
)
ax.add_patch(dedup_box)

ax.text(0.50, flow_y - 0.51, 'Deduplication & Combination', 
        fontsize=12, fontweight='bold', 
        ha='center', va='center', color='white')

# Comparison Section (Bottom)
comp_y = 0.25

ax.text(0.5, comp_y + 0.05, 'SINGLE QUERY vs MULTI-QUERY COMPARISON', 
        fontsize=16, fontweight='bold', 
        ha='center', va='bottom', color=COLORS['header'])

# Single Query Box
single_box = FancyBboxPatch(
    (0.10, comp_y - 0.12), 0.35, 0.12,
    boxstyle="round,pad=0.01", 
    facecolor=COLORS['query1'], 
    edgecolor=COLORS['accent'],
    linewidth=2,
    alpha=0.7,
    zorder=2
)
ax.add_patch(single_box)

ax.text(0.275, comp_y - 0.02, 'SINGLE QUERY', 
        fontsize=14, fontweight='bold', 
        ha='center', va='center', color='white')
ax.text(0.275, comp_y - 0.07, '1 Query → X unique chunks', 
        fontsize=11, 
        ha='center', va='center', color='white')

# VS Text
ax.text(0.5, comp_y - 0.06, 'VS', 
        fontsize=20, fontweight='bold', 
        ha='center', va='center', color=COLORS['secondary'])

# Multi Query Box
multi_box = FancyBboxPatch(
    (0.55, comp_y - 0.12), 0.35, 0.12,
    boxstyle="round,pad=0.01", 
    facecolor=COLORS['success'], 
    edgecolor=COLORS['accent'],
    linewidth=2,
    alpha=0.7,
    zorder=2
)
ax.add_patch(multi_box)

ax.text(0.725, comp_y - 0.02, 'MULTI-QUERY', 
        fontsize=14, fontweight='bold', 
        ha='center', va='center', color='white')
ax.text(0.725, comp_y - 0.07, '4 Queries → Y unique chunks', 
        fontsize=11, 
        ha='center', va='center', color='white')

# Coverage Formula
formula_y = comp_y - 0.20

formula_box = FancyBboxPatch(
    (0.20, formula_y - 0.06), 0.60, 0.06,
    boxstyle="round,pad=0.01", 
    facecolor=COLORS['light'], 
    edgecolor=COLORS['success'],
    linewidth=2,
    zorder=2
)
ax.add_patch(formula_box)

ax.text(0.50, formula_y - 0.03, 
        'Coverage Improvement = (Y - X) / X × 100%', 
        fontsize=13, fontweight='bold', 
        ha='center', va='center', color=COLORS['header'])

# Research Questions Section
questions_y = 0.10

questions_box = FancyBboxPatch(
    (0.05, questions_y - 0.08), 0.90, 0.08,
    boxstyle="round,pad=0.01", 
    facecolor=COLORS['warning'], 
    edgecolor=COLORS['accent'],
    linewidth=2,
    alpha=0.2,
    zorder=1
)
ax.add_patch(questions_box)

ax.text(0.5, questions_y, 'RESEARCH QUESTIONS', 
        fontsize=14, fontweight='bold', 
        ha='center', va='top', color=COLORS['header'])

questions_text = (
    '1. Does multi-query find new relevant chunks or just more noise?  •  '
    '2. What is the token cost trade-off? (3x queries = 3x API calls)  •  '
    '3. Which query types benefit most from expansion?'
)

ax.text(0.5, questions_y - 0.04, questions_text,
        fontsize=10, ha='center', va='top', 
        color=COLORS['dark'])

# Footer
footer_y = 0.02
ax.text(0.5, footer_y, 
        'Repository: github.com/jugalsheth/my-first-rag • ' +
        'Tool: multi_query_rag.py • ' +
        'Tech Stack: ChromaDB + Gemini + Sentence Transformers',
        fontsize=9, style='italic', 
        ha='center', va='top', color=COLORS['dark'])

# Decorative elements
# Top accent bar
accent_bar = Rectangle((0, 0.99), 1, 0.01, 
                      facecolor=COLORS['secondary'], 
                      edgecolor='none', zorder=3)
ax.add_patch(accent_bar)

# Bottom accent bar
accent_bar_bottom = Rectangle((0, 0), 1, 0.01, 
                             facecolor=COLORS['secondary'], 
                             edgecolor='none', zorder=3)
ax.add_patch(accent_bar_bottom)

# Side decorative lines
left_line = Rectangle((0.02, 0.05), 0.005, 0.90, 
                     facecolor=COLORS['secondary'], 
                     edgecolor='none', alpha=0.3, zorder=1)
ax.add_patch(left_line)

right_line = Rectangle((0.98, 0.05), 0.005, 0.90, 
                      facecolor=COLORS['secondary'], 
                      edgecolor='none', alpha=0.3, zorder=1)
ax.add_patch(right_line)

# Set limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Save
plt.tight_layout()
plt.savefig('multiquery_exhibit.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Multi-Query RAG exhibit created: multiquery_exhibit.png")
print("  Size: 16x10 inches at 300 DPI (high resolution)")
print("  Focus: Day 8 Multi-Query RAG experiment")

plt.close()
