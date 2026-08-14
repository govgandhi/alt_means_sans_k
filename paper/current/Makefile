MAIN := main
SUPP := supp
TEX := latexmk
MON_OPT := -pvc
CLN_OPT := -CA

.PHONY: all monitor main-monitor supp supp-monitor wordcount clean

all: ${MAIN}.pdf

${MAIN}.pdf: ${MAIN}.tex references.bib
	${TEX} ${MAIN}.tex

${SUPP}.pdf: ${SUPP}.tex
	${TEX} ${SUPP}.tex

supp: ${SUPP}.pdf

monitor: main-monitor

main-monitor:
	${TEX} ${MON_OPT} ${MAIN}.tex

supp-monitor:
	${TEX} ${MON_OPT} ${SUPP}.tex

clean:
	${TEX} ${CLN_OPT} ${MAIN}.tex
	${TEX} ${CLN_OPT} ${SUPP}.tex

wordcount:
	detex -l -n -w ${MAIN}.tex | wc -w
