# Agent-CodeRAG Research Paper

This directory contains the LaTeX source for the arXiv submission.

## Structure

```
paper/
├── main.tex           — Main manuscript
├── Makefile           — Build automation
├── references.bib     — Bibliography (if needed)
└── figures/           — Architecture diagrams, plots
```

## Quick Start

```bash
cd paper
make          # Compile to PDF
make view     # Open in viewer
make clean    # Clean build artifacts
```

## Dependencies

- LaTeX (TeX Live or MacTeX)
- `latexmk` (for automated builds)
- `pdflatex` with standard packages: `times`, `graphicx`, `amsmath`, `natbib`

## Build

```bash
# First build
make

# Subsequent builds (incremental)
make quick
```

## Submission Checklist

- [ ] Title and authors finalized
- [ ] Abstract < 1920 chars
- [ ] Figures generated (SVG → PDF conversion)
- [ ] Bibliography formatted
- [ ] Compliance with arXiv limits (usually < 100 pages)
- [ ] Source files ZIP for submission

## arXiv Submission

1. Compile: `make` → produces `main.pdf`
2. Gather sources: `main.tex`, `figures/`, `Makefile` (optional)
3. Upload via https://arxiv.org/submit

## Categories

**Primary:** `cs.AI` (Artificial Intelligence)

**Alternatives:**
- `cs.LG` (Machine Learning)
- `cs.SE` (Software Engineering)

## License

MIT © 2026 Igor Boloban