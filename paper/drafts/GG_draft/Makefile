LATEXMK ?= latexmk
LATEXFLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error
SOURCES := preamble.tex authors.tex references.bib

.PHONY: all supp watch clean

all: main.pdf

main.pdf: main.tex $(SOURCES)
	$(LATEXMK) $(LATEXFLAGS) main.tex

supp.pdf: supp.tex $(SOURCES)
	$(LATEXMK) $(LATEXFLAGS) supp.tex

supp: supp.pdf

watch:
	$(LATEXMK) $(LATEXFLAGS) -pvc main.tex

clean:
	$(LATEXMK) -c main.tex supp.tex
