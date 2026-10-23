# LIFESPAN: permanent
"""Which markdown pages the doc tests read, found by looking at the repository and never typed.

A page is in scope when it sits in the repo root, anywhere under docs/ or skills/, or is linked (directly or through
another page in scope) by a relative markdown link that resolves to a file inside the repo. A new page is therefore
covered the day it is added, by the doc tests that loop over `markdown_pages()`.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCANNED_FOLDERS = ("docs", "skills")
# A paragraph about the earlier 20-question run against the live deployment counts a corpus that no longer exists and
# names a claim that was true then, so it is dated on purpose and the doc tests leave it alone.
OTHER_RUN_DATES = ("2026-08-03",)
LINK = re.compile(r"\]\(([^)\s#]+\.md)(?:#[^)]*)?\)")


def markdown_pages(root: Path = ROOT) -> list:
    """Every in-scope markdown page as a path relative to the repo root, sorted."""
    found = {p.name for p in root.glob("*.md")}
    for folder in SCANNED_FOLDERS:
        found |= {p.relative_to(root).as_posix() for p in (root / folder).rglob("*.md")}
    queue = sorted(found)
    while queue:
        page = queue.pop()
        text = (root / page).read_text(encoding="utf-8")
        for target in LINK.findall(text):
            if "://" in target:
                continue
            resolved = (root / page).parent.joinpath(target).resolve()
            try:
                name = resolved.relative_to(root.resolve()).as_posix()
            except ValueError:
                continue  # a link that leaves the repo is not a page of this repo
            if resolved.is_file() and name not in found:
                found.add(name)
                queue.append(name)
    return sorted(found)


def sentences(text: str) -> list:
    """Prose sentences of a markdown file: no code blocks, table rows or images; links and emphasis stripped."""
    text = re.sub(r"```.*?```", "\n\n", text, flags=re.S)
    paragraphs, current = [], []
    for line in text.splitlines():
        s = line.strip()
        starts_new = s.startswith(("- ", "#"))
        if not s or s.startswith(("|", "![", "<")):
            paragraphs.append(" ".join(current))
            current = []
            continue
        if starts_new and current:
            paragraphs.append(" ".join(current))
            current = []
        current.append(s)
    paragraphs.append(" ".join(current))
    out = []
    for p in paragraphs:
        p = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", p).replace("`", "").replace("*", "")
        out.extend(x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", p) if x)
    return out
