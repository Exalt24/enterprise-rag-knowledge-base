# Compare two PDFs against requirements

`app/compare/` reads two PDF documents, checks them against a requirements file and writes a structured QA report. Every finding names the document, the page and the words copied from that page. It is a plain Python package on the same `pypdf` the loader uses, with a command line and one FastAPI route, and it needs no Qdrant, no model and no network.

The documents here are invented: a made-up footbridge design basis (document A) and a made-up calculation report (document B) that is meant to be checked against it. They are in [compare-sample/](compare-sample/), written by `compare-sample/make_samples.py`, and a test checks the committed files are exactly what that script writes.

## Run it

```bash
cd backend
python -m app.compare compare ../docs/compare-sample/design_basis.pdf ../docs/compare-sample/calculation_report.pdf \
    --requirements ../docs/compare-sample/requirements.json --out report.json --text report.txt
python -m app.compare review report.json --reviewer "Dana Planner" --decisions ../docs/compare-sample/decisions.json --out approved.json
python -m app.compare verify approved.json
```

`compare` exits 0, or 1 with `--strict` when a blocking finding is not met, or 2 for a file it cannot use. `review` exits 2 and writes nothing while a blocking finding has no decision. `POST /api/compare` takes `document_a`, `document_b` and `requirements` (the JSON as text) and answers the same report; it has no judge and no approval step.

What the sample run printed (copied from `compare-sample/report.txt`):

```
Document A: design_basis.pdf (2 pages)
Document B: calculation_report.pdf (4 pages, no text on page 3 (not read))

R1  [blocking] DISCREPANCY: Design wind speed agrees with the design basis
    document A says 45 and document B says 40
R3  [blocking] DISCREPANCY: Concrete grade is at least the basis grade
    32 is below the minimum 40
R4  [blocking] DISCREPANCY: The report is not stamped as a draft
    "DRAFT - NOT FOR ISSUE" must not appear and does
R5  [blocking] UNCLEAR: The report records an independent check
    "checked by a second engineer" was not found, but there is no text to read on document B page 3, so it may be there
R8  [major] MISSING: The basis is issued at a named revision
    "Revision:" is not in document A
R9  [minor] UNCLEAR: The report says how the deck was checked for pedestrian vibration
    no judge is configured, so a statement in words is not checked

met 3, discrepancy 3, missing 1, unclear 2; open blocking: R1, R3, R4, R5
```

(The full file also prints the page and the quoted words under each finding, and the three that are met.)

## The requirements file

JSON, `{"requirements": [ ... ]}`. Each entry has an `id`, a `title`, a `kind` and a `severity` (`blocking`, `major` or `minor`).

| kind | what it checks |
|---|---|
| `present` | `phrase` is in `document` (a, b or both) |
| `forbidden` | `phrase` is not in `document` (a draft stamp, a superseded clause) |
| `match` | a `pattern` with a `value` group, and optionally a `unit` group, is read from both documents and the two figures must agree (`tolerance`, default 0) |
| `limit` | the figure read from one document sits inside `min` and/or `max` (inclusive), in `unit` if one is named |
| `judge` | a statement in words that a model judges; it must cite a page and quote words that are really on that page |

The file is validated before any document is read: a pattern that is not a regular expression, a missing `value` group, a `min` above `max`, a duplicate id, an unknown kind or severity are each a clear error.

## The rules that decide a status

- **met**: the documents agree, the figure is within its limit, the phrase is there or the forbidden one is not.
- **discrepancy**: the figures differ, the units differ, a limit is broken, or a forbidden phrase is present.
- **missing**: the item is in no page, and every page of the document could be read.
- **unclear**: the tool cannot say. A page had no text layer, so the item may be on the page nobody read. One document gives two different figures for the same thing (page 2 says 45, page 6 says 40). A figure is not a plain number (`1.200,5`, `1,5`). A model answered without a quote that is on the page it cites, or failed, or there is no model.

A figure is read only as a plain number: `1,200.5` and `1200.5` are the same, and a format that could be read two ways is `unclear` rather than a guess that turns a wrong figure into a pass. A page with no text is listed in the report (`unreadable_pages`), never counted as an empty page, and there is no OCR here.

The model for `judge` requirements is injected. `ChatModelJudge` wraps any chat model that has `invoke` (the service's Groq or Ollama one), tells it the page text is data and not instructions, and parses a JSON verdict; `check.py` accepts `met` or `not_met` only when the quote is on the cited page, so the worst a model can do is make a finding `unclear`. It was run against a test double that returns fenced and bare JSON, invented pages, quotes that are not on the page and failures. It was not run against a live model.

## The approval checkpoint

A report does not clear a package by itself. Every finding that is blocking and not met needs a reviewer's decision (`accept`, `reject` or `waive`) before `review` writes the approved file. A `reject` or `waive` carries a reason of at least ten characters, one decision per finding, none about a finding that is met, and the refusal names every problem at once. The approved file records the reviewer, the decisions and the report's SHA-256, and `verify` recomputes the report's hash and the approval's hash and re-runs the checkpoint, so a changed finding, a changed decision, a renamed reviewer or a rewritten hash is found.

## Tests

```bash
cd backend
python -m pytest tests/test_compare_check.py tests/test_compare_approval.py tests/test_compare_property.py     # 209 tests, no service, no key
python tests/mutation_check_compare.py                                                                     # 61 guards broken one at a time; edits source, so use a copy
```

The property file builds 80 seeded worlds of two generated PDFs each (the same figure present, absent, repeated, contradicted or in a format that is not understood; units that differ; 0 to 2 pages with no text; phrases in random case and broken over two lines; noise lines that look like the requirements) and checks every finding against the status the plan says it must have, and that a quote is words from its page. A coverage test checks the 80 worlds reach every status for every rule that can give it, which is how a generator that never produced an agreeing fee was found and fixed. The mutation check removes each guard in turn (a comparison edge, a boundary, a refusal, the hash) and requires a named test to fail; the first run left one guard covered by the wrong test, and that test was tightened.

## Limits

- No OCR. A scanned page is flagged and makes findings `unclear`; it is not read.
- Matching is on the text a PDF gives, with whitespace collapsed. A table whose cells come out in another order than they read, a figure in a drawing, or a phrase hyphenated at a line end by the author is not found by a phrase search.
- A requirements pattern is a regular expression supplied by the caller: its length is capped, but a pattern written to backtrack badly is not detected.
- A figure is compared as written: no unit conversion (45 m/s and 162 km/h are a discrepancy) and no rounding rules beyond the `tolerance` you set.
- The judge was tested with a double, never with a live model, and a model's `met` is only as good as the quote it cites.
- The sample documents are generated and simple; a real drawing set or a scanned report is not what any test ran on.
