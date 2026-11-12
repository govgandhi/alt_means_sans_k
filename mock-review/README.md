# Mock reviews

Simulated referee reviews of the manuscript, kept so later rounds can be compared with earlier ones.
One folder per round, named `YYYYMMDD`.

Each round has these files:

- `R<n>_<lens>.md`: one review per reviewer. Each reviewer covers a single area of expertise, uses the venue's review form, and ends with an action-item table (`R<n>-NN`).
- `00_META_REVIEW.md`: scores, the problems several reviewers raised, the reviewers' own errors, and a triage plan.
- `01_PR_ISSUE_CANDIDATES.md`: action items grouped into PR and issue candidates. Filed issues and PRs are linked from there.
- `scripts/`: the check scripts the reviewers ran, as they ran them. Their paths point at a temporary scratch directory, so they record what was run and won't rerun as is.

`TRACKER.md` lists every distinct point any round has raised, one row each, with its status:

- **open:** raised, and nothing covers it yet.
- **in progress:** an open PR, branch, or issue an author has taken up covers it.
- **addressed:** fixed in the draft (the row cites where), or its issue was closed as completed.
- **won't address:** the authors decided against it. Set this yourself by editing the row or by closing the point's issue as not planned; the meta-reviewer never sets or undoes it.

Reviewers in later rounds never see earlier rounds or the tracker; the meta-reviewer carries points forward.
Line numbers refer to the commit named in each meta-review.
Reviewers are model-generated and make mistakes; the meta-review lists the ones found.
Leave `mock-review/` out of any anonymized artifact export.
