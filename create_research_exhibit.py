"""
Create Research Exhibit PNG
Generates a visual summary of the RAG research findings
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
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
    'dark': '#2c3e50'
}

# Title Section
title_y = 0.95
ax.text(0.5, title_y, 'RAG RESEARCH EXHIBIT', 
        fontsize=32, fontweight='bold', 
        ha='center', va='top', color=COLORS['header'])
ax.text(0.5, title_y - 0.03, 'Data-Driven Deep Dive into RAG Architecture', 
        fontsize=18, style='italic',
        ha='center', va='top', color=COLORS['accent'])

# Subtitle with date
ax.text(0.5, title_y - 0.06, f'Research Period: January 2026 • Last Updated: {datetime.now().strftime("%B %d, %Y")}',
        fontsize=12, ha='center', va='top', color=COLORS['dark'])

# Timeline Section (Left side)
timeline_x = 0.05
timeline_y_start = 0.85
timeline_height = 0.65

ax.text(timeline_x, timeline_y_start, 'RESEARCH TIMELINE', 
        fontsize=16, fontweight='bold', 
        ha='left', va='top', color=COLORS['header'])

# Timeline items
days = [
    {'day': 'Day 1', 'title': 'Initial Setup', 'desc': 'Basic RAG System\nChromaDB + Embeddings', 'y': 0.80},
    {'day': 'Day 2', 'title': 'Embedding Benchmark', 'desc': 'MPNet (768d) vs\nMiniLM (384d)', 'y': 0.70},
    {'day': 'Day 3', 'title': 'Chunking Experiment', 'desc': 'Laser vs Floodlight\n256/512/1024 tokens', 'y': 0.60},
    {'day': 'Day 4-7', 'title': 'RAGAS Evaluation', 'desc': 'Faithfulness & Relevancy\nSmall chunks win!', 'y': 0.50},
    {'day': 'Day 8', 'title': 'Multi-Query RAG', 'desc': 'Query Expansion\nCoverage Analysis', 'y': 0.40},
]

for day_info in days:
    # Timeline circle
    circle = plt.Circle((timeline_x + 0.02, day_info['y']), 0.008, 
                       color=COLORS['secondary'], zorder=3)
    ax.add_patch(circle)
    
    # Timeline line (connects circles)
    if day_info != days[-1]:
        next_y = days[days.index(day_info) + 1]['y']
        line = plt.Line2D([timeline_x + 0.02, timeline_x + 0.02], 
                         [day_info['y'] - 0.008, next_y + 0.008],
                         color=COLORS['secondary'], linewidth=2, zorder=1)
        ax.add_line(line)
    
    # Day label
    ax.text(timeline_x + 0.04, day_info['y'], day_info['day'], 
            fontsize=12, fontweight='bold', 
            ha='left', va='center', color=COLORS['header'])
    
    # Title
    ax.text(timeline_x + 0.04, day_info['y'] - 0.015, day_info['title'], 
            fontsize=11, ha='left', va='top', color=COLORS['accent'])
    
    # Description
    ax.text(timeline_x + 0.04, day_info['y'] - 0.035, day_info['desc'], 
            fontsize=9, ha='left', va='top', color=COLORS['dark'])

# Key Findings Section (Right side)
findings_x = 0.45
findings_y_start = 0.85

ax.text(findings_x, findings_y_start, 'KEY DISCOVERIES', 
        fontsize=16, fontweight='bold', 
        ha='left', va='top', color=COLORS['header'])

findings = [
    {
        'title': 'Embedding Dimension Impact',
        'finding': 'MPNet (768d) beat MiniLM (384d)\nat scale',
        'icon': '[📊]',
        'y': 0.75
    },
    {
        'title': 'Chunking Paradox',
        'finding': 'Small = Higher similarity\nLarge = Better context',
        'icon': '[🎯]',
        'y': 0.65
    },
    {
        'title': 'RAGAS Verdict',
        'finding': 'Small (256) chunks produced\nmore Faithful answers',
        'icon': '[✓]',
        'y': 0.55
    },
    {
        'title': 'Query Expansion',
        'finding': 'Multi-query improves coverage\n(cost/benefit in progress)',
        'icon': '[🔍]',
        'y': 0.45
    }
]

for finding in findings:
    # Icon box
    icon_box = FancyBboxPatch(
        (findings_x, finding['y'] - 0.04), 0.08, 0.04,
        boxstyle="round,pad=0.01", 
        facecolor=COLORS['success'], 
        edgecolor=COLORS['accent'],
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(icon_box)
    
    ax.text(findings_x + 0.04, finding['y'] - 0.02, finding['icon'], 
            fontsize=12, ha='center', va='center', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['success'], alpha=0.2))
    
    # Finding text
    ax.text(findings_x + 0.10, finding['y'] - 0.008, finding['title'], 
            fontsize=11, fontweight='bold', 
            ha='left', va='top', color=COLORS['header'])
    
    ax.text(findings_x + 0.10, finding['y'] - 0.028, finding['finding'], 
            fontsize=9, ha='left', va='top', color=COLORS['dark'])

# Metrics Section (Center bottom)
metrics_y = 0.30
metrics_x_start = 0.05
metrics_width = 0.90

ax.text(0.5, metrics_y + 0.05, 'EXPERIMENTAL RESULTS', 
        fontsize=16, fontweight='bold', 
        ha='center', va='bottom', color=COLORS['header'])

# Metrics boxes
metrics = [
    {'label': 'Chunking Strategies', 'value': '3', 'unit': 'Small/Medium/Large', 'x': 0.12},
    {'label': 'Embedding Models', 'value': '3', 'unit': 'MiniLM/MPNet/BGE', 'x': 0.35},
    {'label': 'RAGAS Metrics', 'value': '4', 'unit': 'Faith/Rel/Prec/Rec', 'x': 0.58},
    {'label': 'Test Questions', 'value': '5', 'unit': 'Diverse queries', 'x': 0.81},
]

for metric in metrics:
    # Metric box
    metric_box = FancyBboxPatch(
        (metric['x'] - 0.08, metrics_y - 0.05), 0.16, 0.08,
        boxstyle="round,pad=0.01", 
        facecolor=COLORS['light'], 
        edgecolor=COLORS['accent'],
        linewidth=1.5,
        zorder=2
    )
    ax.add_patch(metric_box)
    
    ax.text(metric['x'], metrics_y + 0.02, metric['value'], 
            fontsize=24, fontweight='bold', 
            ha='center', va='center', color=COLORS['secondary'])
    
    ax.text(metric['x'], metrics_y - 0.01, metric['label'], 
            fontsize=9, fontweight='bold', 
            ha='center', va='top', color=COLORS['header'])
    
    ax.text(metric['x'], metrics_y - 0.03, metric['unit'], 
            fontsize=8, 
            ha='center', va='top', color=COLORS['dark'])

# Production Recommendation Section
rec_y = 0.15

rec_box = FancyBboxPatch(
    (0.05, rec_y - 0.08), 0.90, 0.10,
    boxstyle="round,pad=0.01", 
    facecolor=COLORS['success'], 
    edgecolor=COLORS['accent'],
    linewidth=2,
    alpha=0.1,
    zorder=1
)
ax.add_patch(rec_box)

ax.text(0.5, rec_y, 'PRODUCTION RECOMMENDATION (2026)', 
        fontsize=14, fontweight='bold', 
        ha='center', va='top', color=COLORS['header'])

ax.text(0.5, rec_y - 0.025, 
        'Baseline: Small (256 tokens, 20 overlap) OR Medium (512 tokens, 50 overlap) | ' +
        'Embedding: all-mpnet-base-v2 (768d) | ' +
        'Evaluation: RAGAS (Faithfulness + Answer Relevancy)',
        fontsize=10, ha='center', va='top', 
        color=COLORS['dark'], wrap=True)

# Footer
footer_y = 0.03
ax.text(0.5, footer_y, 
        'Repository: github.com/jugalsheth/my-first-rag • ' +
        'Methodology: Data-Driven RAG Research • ' +
        'Tools: ChromaDB + LangChain + RAGAS + Gemini',
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
left_line = Rectangle((0.02, 0.1), 0.005, 0.75, 
                     facecolor=COLORS['secondary'], 
                     edgecolor='none', alpha=0.3, zorder=1)
ax.add_patch(left_line)

right_line = Rectangle((0.98, 0.1), 0.005, 0.75, 
                      facecolor=COLORS['secondary'], 
                      edgecolor='none', alpha=0.3, zorder=1)
ax.add_patch(right_line)

# Set limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Save
plt.tight_layout()
plt.savefig('research_exhibit.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Research exhibit created: research_exhibit.png")
print("  Size: 16x10 inches at 300 DPI (high resolution)")
print("  Perfect for presentations, documentation, or research papers")

plt.close()
