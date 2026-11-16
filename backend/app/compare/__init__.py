"""Compare two PDF documents against predefined requirements and report what does not agree, with the page and the words behind every finding.

    pages.py         read a PDF into numbered pages, and list the pages that gave no text
    requirements.py  the requirements file: present, forbidden, match, limit and judge
    check.py         the rules that turn pages and requirements into findings
    judge.py         a judge for statements in words, on any chat model, whose answer is accepted only with a quote that is on the page it cites
    report.py        the structured report, its hash and its plain text form
    approval.py      the human approval checkpoint, bound to the report's hash
    api.py           the FastAPI route; __main__.py is the command line

Method, limits and test counts: docs/compare.md.
"""
from .approval import approve, verify_approved
from .check import compare
from .judge import ChatModelJudge
from .pages import CompareError, read_pdf
from .report import Report, render_text
from .requirements import load_requirements, parse_requirements

__all__ = ["approve", "verify_approved", "compare", "ChatModelJudge", "CompareError", "read_pdf", "Report", "render_text", "load_requirements", "parse_requirements"]
