"""
Retrieval accuracy evaluation.

Measures how often the chunk that actually contains the answer appears in the retrieved
set, across the three strategies. This is RETRIEVAL accuracy only, deliberately separate
from answer quality: generation is non-deterministic and would blur what is being measured.

Scoring is hit@k. Each question names a phrase that appears verbatim only in the correct
chunk; a hit means at least one retrieved source contains it. Anchor phrases are checked
against the corpus before scoring, so a typo in the eval set fails loudly instead of
silently reporting 0%.

Run against a deployed instance:
    python tests/eval_retrieval.py --base-url https://enterprise-rag-api.onrender.com/api
"""

import argparse
import json
import time
import urllib.request


# (question, phrase that appears ONLY in the chunk that answers it)
EVAL_SET = [
    ("What accuracy does basic vector search achieve?", "about 40 percent"),
    ("How is hybrid search weighted between vector and keyword?", "70 percent"),
    ("What score does cross-encoder reranking reach?", "67.7 percent"),
    ("Which embedding model is used?", "all-MiniLM-L6-v2"),
    ("How many dimensions do the embeddings have?", "384-dimensional"),
    ("Why was MiniLM chosen despite being old?", "512MB memory ceiling"),
    ("How large are the chunks and how much do they overlap?", "500-character chunks"),
    ("Why does chunk overlap matter?", "spanning a chunk boundary"),
    ("Which vector database is used and what was it migrated from?", "migrated from Chroma"),
    ("Why is normalization enabled?", "assumes normalized vectors"),
    ("What happens if Ollama times out?", "falls back to Groq"),
    ("What temperature does generation run at?", "temperature 0.1"),
    ("What is sent before the first token when streaming?", "first frame"),
    ("What marks the end of an OpenAI-compatible stream?", "[DONE]"),
    ("Why does streaming skip the cache?", "synthetic tokens"),
    ("What status code is returned when the vector store is down?", "return 503"),
    ("What does the health endpoint report separately?", "vector store connectivity"),
    ("What is included in the cache key?", "retrieval options"),
    ("What memory limit does the backend run under?", "512MB memory limit"),
    ("Where is the frontend deployed?", "Vercel"),
]

STRATEGIES = [
    ("vector only",      {"use_hybrid_search": False, "use_reranking": False}),
    ("hybrid (BM25+vec)", {"use_hybrid_search": True,  "use_reranking": False}),
    ("hybrid + rerank",  {"use_hybrid_search": True,  "use_reranking": True}),
]


def post(url, payload, timeout=300):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def load_corpus(base_url):
    """
    Full text of every stored chunk, keyed by the preview prefix the API returns.

    Necessary because /query only returns a 100-character content_preview. Scoring the
    anchor against that preview would measure "does the phrase fall in the first 100
    characters", not "was the correct chunk retrieved", and would badly understate accuracy.
    """
    docs = json.loads(urllib.request.urlopen(f"{base_url}/documents", timeout=180).read())
    chunks = []
    for d in docs.get("documents", docs if isinstance(docs, list) else []):
        text = d.get("page_content") or d.get("content") or ""
        if text:
            chunks.append(text)
    return chunks


def resolve(preview, chunks):
    """Map a truncated preview back to its full chunk text."""
    stem = preview.replace("...", "")[:60]
    for c in chunks:
        if c.startswith(stem[:40]) or stem[:40] in c:
            return c
    return preview


def retrieved_text(base_url, question, k, opts, chunks):
    """Return the FULL text of every chunk retrieved for a question."""
    data = post(f"{base_url}/query", {"question": question, "k": k, **opts})
    return " ".join(
        resolve(s.get("content_preview", ""), chunks) for s in data.get("sources", [])
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8001/api")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    print(f"Corpus check against {args.base_url}")
    stats = json.loads(
        urllib.request.urlopen(f"{args.base_url}/stats", timeout=180).read()
    )
    print(f"  documents indexed: {stats.get('total_documents')}")
    print(f"  k = {args.k}, questions = {len(EVAL_SET)}")

    chunks = load_corpus(args.base_url)
    print(f"  chunks fetched   : {len(chunks)}")

    # Fail loudly on a bad eval set rather than silently scoring it 0%.
    corpus = " ".join(chunks).lower()
    bad = [a for _, a in EVAL_SET if a.lower() not in corpus]
    if bad:
        print(f"\n  [!] {len(bad)} anchor(s) absent from the corpus, these can never hit:")
        for a in bad:
            print(f"      - {a}")
        print("      Fix the eval set before trusting the numbers.\n")
    else:
        print("  all anchors present in corpus\n")

    results = {}
    for label, opts in STRATEGIES:
        hits, misses = 0, []
        t0 = time.time()
        for question, anchor in EVAL_SET:
            try:
                found = anchor.lower() in retrieved_text(
                    args.base_url, question, args.k, opts
                ).lower()
            except Exception as e:
                print(f"  [!] {question[:40]}... errored: {e}")
                found = False
            if found:
                hits += 1
            else:
                misses.append(anchor)
        pct = 100.0 * hits / len(EVAL_SET)
        results[label] = pct
        print(f"{label:20} hit@{args.k}: {hits}/{len(EVAL_SET)} = {pct:.1f}%"
              f"   ({time.time() - t0:.0f}s)")
        if misses:
            print(f"{'':20} missed: {', '.join(m[:28] for m in misses[:5])}")

    print("\nSummary")
    for label, pct in results.items():
        print(f"  {label:20} {pct:.1f}%")


if __name__ == "__main__":
    main()
