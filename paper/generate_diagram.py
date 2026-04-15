#!/usr/bin/env python3
"""Generate architecture diagram SVG for Agent-CodeRAG paper."""

import svgwrite

# Create SVG drawing
dwg = svgwrite.Drawing('architecture.svg', size=('800px', '500px'), profile='tiny')

# Styles
STYLE_NODE = {
    'fill': '#e1f5fe',
    'stroke': '#0288d1',
    'stroke_width': 2,
    'rx': 8,
    'ry': 8,
}
STYLE_TEXT = {'font-family': 'Arial, sans-serif', 'font-size': '11px', 'fill': '#333'}
STYLE_ARROW = {'stroke': '#555', 'stroke_width': 2, 'fill': 'none', 'marker-end': 'url(#arrow)'}

# Define nodes: (label, x, y, width, height)
nodes = [
    ("Local Python Code", 50, 220, 140, 40),
    ("AST Parser", 250, 220, 110, 40),
    ("Delta-Sync", 420, 220, 100, 40),
    ("LLM Distiller", 580, 130, 120, 40),
    ("Local Cache", 580, 310, 110, 40),
    ("Semantic Summary", 750, 220, 140, 40),
    ("ONNX Embedder", 950, 220, 110, 40),
    ("DuckDB VSS", 1120, 220, 100, 40),
    ("Semantic Search /\nJSON API", 1290, 220, 140, 50),
]

# Draw nodes
for label, x, y, w, h in nodes:
    dwg.add(dwg.rect(insert=(x, y), size=(w, h), **STYLE_NODE))
    lines = label.split('\n')
    for i, line in enumerate(lines):
        offset = (i - (len(lines)-1)/2) * 14
        dwg.add(dwg.text(line, insert=(x + w/2, y + h/2 + offset),
                        text_anchor='middle', **STYLE_TEXT))

# Define arrow marker
dwg.defs.add(dwg.marker(id='arrow', orient='auto', size=(10, 7),
                        viewBox='0 0 10 10',
                        path=svgwrite.path.Path('M 0 0 L 10 5 L 0 10').d()))
# Arrows (start_x, start_y, end_x, end_y)
arrows = [
    # Local Python Code -> AST Parser
    (190, 240, 250, 240),
    # AST Parser -> Delta-Sync
    (360, 240, 420, 240),
    # Delta-Sync -> LLM Distiller (changed)
    (480, 240, 580, 170),
    # Delta-Sync -> Local Cache (unchanged)
    (480, 260, 580, 310),
    # LLM Distiller -> Semantic Summary
    (640, 150, 750, 240),
    # Local Cache -> Semantic Summary
    (640, 310, 750, 260),
    # Semantic Summary -> ONNX Embedder
    (890, 240, 950, 240),
    # ONNX Embedder -> DuckDB VSS
    (1060, 240, 1120, 240),
    # DuckDB VSS -> Semantic Search
    (1220, 240, 1290, 245),
]

for x1, y1, x2, y2 in arrows:
    dwg.add(dwg.line(start=(x1, y1), end=(x2, y2), **STYLE_ARROW))

dwg.save()
print("Saved architecture.svg")
