.PHONY: all clean build view

TEX = main.tex
PDF = main.pdf

all: $(PDF)

$(PDF): $(TEX)
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" $<

clean:
	latexmk -C
	rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk

view: $(PDF)
	open $(PDF)  # macOS
	# xdg-open $(PDF) on Linux
	# start $(PDF) on Windows

# Quick compile for editing
quick:
	pdflatex main.tex