"""No markdown page of the repo may keep a claim that stopped being true.

docs/SYSTEM-KNOWLEDGE.md once said "There is no CI test run" and described a 20-question script, long after CI ran the
pinned tests, a 42-question evaluation and a mutation check. The count test only read three pages, so the page went stale
unseen, and the README linked to it as the design notes. Two guards now cover the class:

1. The set of pages is found by looking at the repo (tests/doc_pages.py: the root, docs/, skills/ and anything they link),
   never typed, so a page added tomorrow is covered tomorrow. Its own tests build a throwaway tree to prove the finding.
2. Every page is checked against a list of claims that used to be true (and the counted quantities in
   test_eval_readme_counts.py, which reads the same page set). A pattern is proved on a sample sentence, so a typo in a
   regex cannot make a page pass by matching nothing. A paragraph dated to the earlier run (OTHER_RUN_DATES) is a record
   and is left alone.

When a page is allowed to say something new and true that one of these patterns catches, change the pattern, not the page.
Run: pytest tests/test_eval_docs_not_stale.py
"""
import itertools
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

from doc_pages import OTHER_RUN_DATES, ROOT, SCANNED_FOLDERS, markdown_pages  # noqa: E402

PAGES = markdown_pages()

# claim that used to be true -> (pattern, a sentence it must catch)
STALE_CLAIMS = {
    "no CI": (r"\bno CI\b", "There is no CI test run."),
    "not run in CI": (r"\bnot (?:run|tested|checked|covered) (?:in|by) CI\b", "The tests are not run in CI."),
    "wire the tests into CI": (r"\bwire the tests into CI\b", "What I would change: wire the tests into CI."),
    "twenty questions": (r"\b(?:20|twenty)[ -]questions\b", "A script that scores 20 questions by hit@k."),
    "no automated tests": (r"\bno (?:automated )?test (?:suite|run)\b|\bno automated tests?\b", "The repo has no automated tests."),
    "tests only by hand": (r"\btests (?:are )?(?:only )?run (?:only )?by hand\b", "The tests are run by hand."),
    "no continuous integration": (r"\bno (?:continuous integration|CI pipeline|CI workflow)\b|\bwithout CI\b", "It has no CI pipeline."),
    "CI does not exist": (r"\bCI (?:is|has) not (?:set up|configured|wired)\b", "CI is not set up yet."),
}

# what the page that describes testing and the architecture must say today, so it cannot drift back by saying nothing
MUST_SAY = {
    "docs/SYSTEM-KNOWLEDGE.md": ("ci.yml", "mutation_check", "run_eval.py --check", "mcp_server/server.py", "skills/querying-knowledge-base"),
    "README.md": ("MCP server", "mutation check", "CI"),
}


def paragraphs(page):
    """The page as flat paragraphs (emphasis and code ticks dropped), minus those about the earlier dated run."""
    text = (ROOT / page).read_text(encoding="utf-8").replace("`", "").replace("*", "")
    flat = (" ".join(p.split()) for p in re.split(r"\n\s*\n", text))
    return [p for p in flat if p and not any(date in p for date in OTHER_RUN_DATES)]


# CLASS-TABLE: every markdown page of the repo (found by globbing) x every claim that used to be true; none may appear
STALE_CASES = list(itertools.product(PAGES, STALE_CLAIMS))


@pytest.mark.parametrize("page,claim", STALE_CASES, ids=lambda v: v)
def test_no_page_keeps_a_claim_that_stopped_being_true(page, claim):
    pattern, _ = STALE_CLAIMS[claim]
    hits = [p[:140] for p in paragraphs(page) if re.search(pattern, p, flags=re.I)]
    assert not hits, f"{page} says {claim!r} and the repo no longer matches it: {hits[0]!r}. Rewrite it to what is true now."


@pytest.mark.parametrize("claim", STALE_CLAIMS, ids=lambda v: v)
def test_each_stale_claim_pattern_catches_its_sample(claim):
    pattern, sample = STALE_CLAIMS[claim]
    assert re.search(pattern, sample, flags=re.I), f"the pattern for {claim!r} no longer matches its own sample sentence"


# CLASS-TABLE: page that describes testing or architecture x what it must say today
SAY_CASES = [(page, phrase) for page, phrases in MUST_SAY.items() for phrase in phrases]


@pytest.mark.parametrize("page,phrase", SAY_CASES, ids=lambda v: v)
def test_the_page_says_what_is_true_today(page, phrase):
    assert page in PAGES, f"{page} is no longer a page of the repo: update MUST_SAY"
    assert phrase in (ROOT / page).read_text(encoding="utf-8"), f"{page} no longer mentions {phrase!r}"


# ---------------------------------------------------------------- the page set is found, not typed


def test_every_markdown_file_under_the_scanned_folders_is_a_page():
    on_disk = {p.relative_to(ROOT).as_posix() for folder in SCANNED_FOLDERS for p in (ROOT / folder).rglob("*.md")}
    assert on_disk <= set(PAGES), f"pages missing from the scan: {sorted(on_disk - set(PAGES))}"
    assert "README.md" in PAGES


def test_a_linked_page_and_a_new_page_are_found_and_a_page_outside_the_repo_is_not(tmp_path):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "notes").mkdir()
    (tmp_path / "outside.md").write_text("# outside\n", encoding="utf-8")
    (repo / "README.md").write_text("[a](notes/linked.md#top) [b](https://example.com/x.md) [c](../outside.md)\n", encoding="utf-8")
    (repo / "notes" / "linked.md").write_text("[d](deeper.md)\n", encoding="utf-8")
    (repo / "notes" / "deeper.md").write_text("# deeper\n", encoding="utf-8")
    (repo / "notes" / "unlinked.md").write_text("# nobody links here\n", encoding="utf-8")
    (repo / "docs" / "brand-new.md").write_text("# added today\n", encoding="utf-8")
    assert markdown_pages(repo) == ["README.md", "docs/brand-new.md", "notes/deeper.md", "notes/linked.md"]
