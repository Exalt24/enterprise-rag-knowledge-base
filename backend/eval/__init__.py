"""Retrieval evaluation harness: synthetic corpus, labelled questions, deterministic embedder, baseline.

RENDER=true is set before any app module is imported. advanced_retrieval otherwise imports sentence_transformers
(torch) and builds an Ollama client at import, neither of which the evaluation uses: it never reranks and never calls
a model. setdefault leaves an explicit RENDER alone.
"""
import os

os.environ.setdefault("RENDER", "true")
