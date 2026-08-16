MAIN := main
SUPP := supp
NOTE := note
TEX := latexmk
MON_OPT := -pvc
CLN_OPT := -CA

.PHONY: all main-monitor supp-monitor note-monitor wordcount clean

all: ${MAIN}.pdf ${SUPP}.pdf ${NOTE}.pdf

${MAIN}.pdf:
	${TEX} ${MAIN}.tex

${SUPP}.pdf:
	${TEX} ${SUPP}.tex

${NOTE}.pdf:
	${TEX} ${NOTE}.tex

main-monitor:
	${TEX} ${MON_OPT} ${MAIN}.tex

supp-monitor:
	${TEX} ${MON_OPT} ${SUPP}.tex

note-monitor:
	${TEX} ${MON_OPT} ${NOTE}.tex

clean:
	${TEX} ${CLN_OPT} ${MAIN}.tex
	${TEX} ${CLN_OPT} ${SUPP}.tex
	${TEX} ${CLN_OPT} ${NOTE}.tex

wordcount:
	detex -l -n -w ${MAIN}.tex | wc -w

