MAIN := main
SUPP := supp
NOTE := note
TEX := latexmk
MON_OPT := -pvc
CLN_OPT := -CA

.PHONY: all monitor main-monitor supp supp-monitor note-monitor norm-geometry wordcount clean

all: ${MAIN}.pdf ${SUPP}.pdf ${NOTE}.pdf

${MAIN}.pdf: ${MAIN}.tex references.bib ${SUPP}.pdf
	${TEX} -pdf -interaction=nonstopmode -halt-on-error ${MAIN}.tex

${SUPP}.pdf: ${SUPP}.tex factor_balance.tex references.bib ../../results/prospective_baselines/table.tex ../../results/prospective_baselines/balanced_table.tex ../../results/factor_balance/table.tex
	${TEX} -pdf -interaction=nonstopmode -halt-on-error ${SUPP}.tex

${NOTE}.pdf:
	${TEX} ${NOTE}.tex

supp: ${SUPP}.pdf

norm-geometry:
	${TEX} -pdf -interaction=nonstopmode -halt-on-error norm_geometry_supplement.tex

monitor: main-monitor

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
