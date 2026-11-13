# Author's working draft

Start writing in `main.tex`, below `\maketitle`. The abstract and body are blank;
the headings are suggestions you can freely rename or remove. Author information
was copied from the current manuscript into `authors.tex`.

This folder is an independent draft. `paper/current/` remains the current
manuscript. Nothing here is automatically promoted, synchronized, or uploaded.
When you decide this draft should become the main manuscript, archive the old
current version, move this draft into `paper/current/`, and handle upload then.

## Files and building

- `main.tex`: your paper, with PRE formatting and the complete reference list.
- `references.bib`: the full existing bibliography, consolidated to 20 distinct
  works, plus seven papers identified during the October 27 audit.
- `authors.tex`: shared author names, affiliations, and corresponding email.
- `preamble.tex`: packages, math commands, and theorem environments.
- `supp.tex`: optional separate space for supporting material.

From the repository root:

```sh
make -C paper/drafts/GG_draft
make -C paper/drafts/GG_draft supp
make -C paper/drafts/GG_draft watch
```

The first command produces `main.pdf`; the second produces `supp.pdf`; the third
rebuilds as you save edits. `make -C paper/drafts/GG_draft clean` removes build
auxiliaries while keeping PDFs. Dependencies are a TeX installation containing
REVTeX 4.2, pdfLaTeX, BibTeX, and latexmk. Initial build checked with TeX Live 2025
and latexmk 4.86a. No files from `paper/current/` are needed to build this folder.

## References while writing

Use, for example, `...~\cite{murray2023unsupervised,oyama2022norm}`. Search
`references.bib` for a title to find its citation key. Add new entries directly
to that file. `\nocite{*}` in `main.tex` deliberately prints every reference,
even before you have cited it. Remove that line when preparing a submission.

The original bibliography had 24 records representing 20 distinct works. All
works are retained; duplicate/preprint aliases were consolidated as follows:

| Old alias | Key retained in this draft |
|---|---|
| `schakel2015` | `schakel2015measuring` |
| `murray2023` | `murray2023unsupervised` |
| `kojaku2023` | `kojaku2024network` |
| `oyama2023` | `oyama2022norm` |

The key `oyama2022norm` is retained for compatibility; the entry correctly lists
the 2023 EMNLP publication. Existing metadata was carried over, not subjected to
a fresh comprehensive reference audit. Source URLs for the seven additions are
recorded next to their BibTeX entries. Harker and Bhaskara (2024) and Karkada et
al. (2026) use their published ICML records, which are now verified; the earlier
audit discussed their arXiv versions. Su et al. (2026) remains a preprint entry.

## Target journal

Target: **Physical Review E**, collection **Network Science for Understanding
AI**, with inviting editors Robert Jankowski, Filippo Radicchi, and Bruno
Loureiro, as specified by you on October 29, 2026. The collection information is
recorded here for later submission; it is not a paper title or an author list.

The scaffold follows [PRE's author guidance](https://journals.aps.org/pre/authors)
and [APS REVTeX guidance](https://journals.aps.org/revtex), checked October 29,
2026: `revtex4-2`, `aps,pre,reprint`, and `apsrev4-2` bibliography style.
`longbibliography` includes article titles for convenient reading. There are no
custom margin or font overrides. Change `reprint` to `preprint` in `main.tex`
for the supported single-column editing layout.

The public collection-specific call was not located during setup, so no
collection-specific deadline or additional formatting rule is assumed. The
journal format is in place; submission details can be checked when the draft
is ready.
