"""Draft a quotation from an enquiry, a fee schedule, business rules and a template, with every doubt flagged and a reviewer's approval bound to the draft's hash.

Method, limits and a sample run: docs/quote.md. The command line is `python -m app.quote`.
"""
from .quote import Quote, approve, draft, extract_with_model, parse_enquiry, parse_schedule, verify

__all__ = ["Quote", "approve", "draft", "extract_with_model", "parse_enquiry", "parse_schedule", "verify"]
