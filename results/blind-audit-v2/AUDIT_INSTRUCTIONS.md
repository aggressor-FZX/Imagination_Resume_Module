# Blind semantic audit

Judge each document in `documents/` (27 files) on its own merits. Do **not** open
`DO_NOT_OPEN_key.json` or `meta/` — those identify which configuration produced
each document, and the point of this audit is that the judgement is about the
document, not the configuration.

The documents are byte-identical copies. Nothing has been removed, reordered or
reworded, so any awkward phrasing you see is the system's own.

For every sentence that differs from the source, record one verdict:

- **FAITHFUL** — same fact, different wording. A novel word is not an invented fact.
- **FAITHFUL (capability)** — a general capability generalised from a demonstrated
  activity, which the Imaginative policy permits.
- **ADDITION** — states something the sources do not: a method, tool, credential,
  date, metric or relationship the candidate never had.
- **MISATTRIBUTED** — a real fact attached to the wrong employer or project.
- **OMITTED** — a source-backed fact the document drops.

Then answer, for the document as a whole: **would you deliver this?** yes / no, and
in one sentence why.

