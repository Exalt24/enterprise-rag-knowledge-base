"""Library behind run_eval.py: corpus, labelled questions, in-memory store, retrieval runs, metrics, baseline.

Everything goes through the repo's own code: TextChunker for chunking, the real VectorStoreService (its search, scroll
and stats code) backed by a QdrantVectorStore on QdrantClient(":memory:"), retrieval_service for vector search and
AdvancedRetrieval for BM25 and hybrid. Only the embedder is swapped, and only at the point where the repo builds it.

Labels are CHUNK level: a question carries a must_contain phrase copied verbatim from the corpus, and a retrieved chunk
counts only if its text contains that phrase. A test proves each phrase sits in exactly the document the question names.
"""
import contextlib
import hashlib
import io
import json
import math
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import settings
from app.services.advanced_retrieval import AdvancedRetrieval
from app.services.chunking import TextChunker
from app.services.retrieval import retrieval_service
from app.services.vector_store import vector_store

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
CALIBRATION_PATH = EVAL_DIR / "calibration.jsonl"
BASELINE_PATH = EVAL_DIR / "baseline.json"

CATEGORIES = ("direct", "paraphrase", "multi-document", "unanswerable")
METHODS = ("vector", "bm25", "hybrid")
ANSWERABLE_METRICS = ("hit@1", "hit@3", "mrr", "recall@3", "recall@5", "kept")
UNANSWERABLE_METRICS = ("abstain",)
RETRIEVAL_DEPTH = 5  # chunks kept per query; recall@5 and MRR are measured inside this window
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20261023
BASELINE_TOLERANCE = 0.02  # a deterministic number may fall this far (absolute) before the check fails
_ID_NAMESPACE = uuid.UUID("5d0e8a56-3b0e-4c7e-9f64-2f1f1b9d0a11")


@contextlib.contextmanager
def quiet():
    """The services print to stdout on every call; keep a report readable by swallowing that."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# --------------------------------------------------------------------------- questions and corpus


@dataclass(frozen=True)
class Need:
    """One chunk a question needs: the document it lives in and a verbatim phrase from it."""

    document: str
    phrase: str


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    text: str
    needs: Tuple[Need, ...]  # empty for an unanswerable question


def load_questions(path: Path = QUESTIONS_PATH) -> List[Question]:
    questions = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        needs = []
        if row["must_contain"]:
            needs.append(Need(row["expected_document"], row["must_contain"]))
        for extra in row.get("also", []):
            needs.append(Need(extra["expected_document"], extra["must_contain"]))
        questions.append(Question(row["id"], row["category"], row["question"], tuple(needs)))
    return questions


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> List[Document]:
    """One Document per .md file, sorted by name. read_text normalises CRLF, so the chunk hash is the same on every OS."""
    docs = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        docs.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"file_name": path.name, "file_type": "md", "source": path.name},
            )
        )
    return docs


def chunk_corpus(docs: Sequence[Document]) -> List[Document]:
    """Chunk with the repo's own chunker and give each chunk a stable id, so scroll order and BM25 ties do not change."""
    chunks = TextChunker.chunk_documents(list(docs))
    for chunk in chunks:
        key = f"{chunk.metadata['file_name']}#{chunk.metadata['chunk_index']}"
        chunk.id = str(uuid.uuid5(_ID_NAMESPACE, key))
    return chunks


def chunk_fingerprint(chunks: Sequence[Document]) -> str:
    """Hash of every chunk's file, index and text: it changes when the splitter changes, and only then."""
    payload = json.dumps(
        [[c.metadata["file_name"], c.metadata["chunk_index"], c.page_content] for c in chunks], ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def phrase_in_text(phrase: str, text: str) -> bool:
    """Chunk-level relevance: the chunk text holds the verbatim phrase. An empty phrase matches nothing."""
    return bool(phrase) and phrase in text


def chunk_satisfies(need: Need, chunk: Document) -> bool:
    return chunk.metadata.get("file_name") == need.document and phrase_in_text(need.phrase, chunk.page_content)


def abstains(top_score: float, threshold: float) -> bool:
    """The system declines to answer when even its best chunk scores below the calibrated threshold."""
    return top_score < threshold


# --------------------------------------------------------------------------- store and retrieval


@contextlib.contextmanager
def in_memory_store(embeddings, chunks: Sequence[Document]):
    """Point the repo's VectorStoreService at QdrantClient(":memory:") and ingest `chunks` through its add_documents.

    The service is a process singleton, so its two connection attributes are restored afterwards.
    """
    client = QdrantClient(":memory:")
    dim = len(embeddings.embed_query("dimension probe"))
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    saved = (vector_store._client, vector_store._vectorstore)
    vector_store._client = client
    vector_store._vectorstore = QdrantVectorStore(
        client=client, collection_name=settings.qdrant_collection, embedding=embeddings
    )
    try:
        with quiet():
            vector_store.add_documents(list(chunks))
        yield vector_store
    finally:
        vector_store._client, vector_store._vectorstore = saved
        client.close()


@dataclass
class Retrieved:
    chunks: List[Document]
    top_score: float


class Retrievers:
    """The three retrieval modes, each going through the repo's code. One AdvancedRetrieval per run, so its BM25 cache is per corpus."""

    def __init__(self, depth: int = RETRIEVAL_DEPTH):
        self.depth = depth
        self.engine = AdvancedRetrieval()

    @property
    def fallbacks(self) -> int:
        return self.engine.hybrid_fallback_count

    def vector(self, query: str) -> Retrieved:
        with quiet():
            result = retrieval_service.retrieve(query, k=self.depth, with_scores=True)
        return Retrieved(list(result.documents), float(result.scores[0]) if result.scores else 0.0)

    def bm25(self, query: str) -> Retrieved:
        with quiet():
            retriever = self.engine._get_bm25_retriever(self.depth)
            docs = retriever.invoke(query)
            scores = retriever.vectorizer.get_scores(retriever.preprocess_func(query))
        return Retrieved(list(docs), float(max(scores)) if len(scores) else 0.0)

    def hybrid(self, query: str) -> Retrieved:
        with quiet():
            docs, scores = self.engine.hybrid_search(query, k=self.depth, with_scores=True)
        return Retrieved(list(docs), float(scores[0]) if scores else 0.0)

    def by_name(self, method: str) -> Callable[[str], Retrieved]:
        return {"vector": self.vector, "bm25": self.bm25, "hybrid": self.hybrid}[method]


# --------------------------------------------------------------------------- scoring


def need_ranks(question: Question, retrieved: Retrieved) -> List[Optional[int]]:
    """For each need, the 1-based rank of the first retrieved chunk that satisfies it, or None."""
    ranks = []
    for need in question.needs:
        rank = None
        for position, chunk in enumerate(retrieved.chunks, 1):
            if chunk_satisfies(need, chunk):
                rank = position
                break
        ranks.append(rank)
    return ranks


def question_metrics(question: Question, retrieved: Retrieved, threshold: float) -> Dict[str, float]:
    declined = abstains(retrieved.top_score, threshold)
    if question.category == "unanswerable":
        return {"abstain": 1.0 if declined else 0.0}
    ranks = need_ranks(question, retrieved)
    found = [r for r in ranks if r is not None]

    def hit(k):
        return 1.0 if any(r <= k for r in found) else 0.0

    def recall(k):
        return sum(1 for r in found if r <= k) / len(ranks)

    return {
        "hit@1": hit(1),
        "hit@3": hit(3),
        "mrr": 1.0 / min(found) if found else 0.0,
        "recall@3": recall(3),
        "recall@5": recall(5),
        "kept": 0.0 if declined else 1.0,
    }


def bootstrap_interval(values: Sequence[float], rng: random.Random, resamples: int = BOOTSTRAP_RESAMPLES) -> Tuple[float, float]:
    """95% percentile interval of the mean, resampling questions with replacement."""
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def chance_hit(n_chunks: int, n_relevant: int, k: int) -> float:
    """Chance that k chunks drawn at random contain at least one of the n_relevant chunks."""
    k = min(k, n_chunks)
    return 1.0 - math.comb(n_chunks - n_relevant, k) / math.comb(n_chunks, k)


def chance_mrr(n_chunks: int, n_relevant: int, depth: int) -> float:
    """Expected reciprocal rank of the first relevant chunk in a random ordering, counted inside the depth window."""
    total = 0.0
    for rank in range(1, min(depth, n_chunks) + 1):
        total += math.comb(n_chunks - rank, n_relevant - 1) / math.comb(n_chunks, n_relevant) / rank
    return total


def calibrate_threshold(answerable_scores: Sequence[float], unanswerable_scores: Sequence[float]) -> float:
    """Pick the abstain threshold that best separates two score lists, from a set kept apart from the eval questions.

    Candidates are the midpoints between neighbouring distinct scores. The score is balanced accuracy: the share of
    answerable questions kept plus the share of unanswerable ones declined, halved. The lowest best candidate wins a tie.
    """
    pooled = sorted(set(answerable_scores) | set(unanswerable_scores))
    if len(pooled) < 2:
        return pooled[0] if pooled else 0.0
    best, best_score = pooled[0], -1.0
    for low, high in zip(pooled, pooled[1:]):
        candidate = (low + high) / 2.0
        kept = sum(1 for s in answerable_scores if not abstains(s, candidate)) / len(answerable_scores)
        declined = sum(1 for s in unanswerable_scores if abstains(s, candidate)) / len(unanswerable_scores)
        score = (kept + declined) / 2.0
        if score > best_score + 1e-12:
            best, best_score = candidate, score
    return best


def run_evaluation(
    embeddings,
    questions: Sequence[Question],
    calibration: Sequence[Question],
    chunks: Sequence[Document],
) -> dict:
    """Ingest `chunks`, run every method over the calibration and eval questions, and return the full report."""
    rng = random.Random(BOOTSTRAP_SEED)
    with in_memory_store(embeddings, chunks) as service:
        stats = service.get_stats()  # through the real get_stats code
        retrievers = Retrievers()
        thresholds, outcomes = {}, {}
        for method in METHODS:
            run = retrievers.by_name(method)
            answerable, unanswerable = [], []
            for q in calibration:
                (unanswerable if q.category == "unanswerable" else answerable).append(run(q.text).top_score)
            thresholds[method] = calibrate_threshold(answerable, unanswerable)
            outcomes[method] = [(q, run(q.text)) for q in questions]
        fallbacks = retrievers.fallbacks

    results = {}
    for method in METHODS:
        results[method] = {}
        for category in CATEGORIES:
            rows = [question_metrics(q, r, thresholds[method]) for q, r in outcomes[method] if q.category == category]
            if not rows:
                continue
            cell = {}
            for metric in rows[0]:
                values = [row[metric] for row in rows]
                lo, hi = bootstrap_interval(values, rng)
                cell[metric] = {"value": sum(values) / len(values), "lo": lo, "hi": hi}
            results[method][category] = {"n": len(rows), **cell}

    n_chunks = len(chunks)
    chance = {}
    for category in CATEGORIES:
        if category == "unanswerable":
            continue
        relevant = [
            sum(1 for c in chunks if any(chunk_satisfies(n, c) for n in q.needs))
            for q in questions
            if q.category == category
        ]
        if not relevant:
            continue
        chance[category] = {
            "hit@1": sum(chance_hit(n_chunks, m, 1) for m in relevant) / len(relevant),
            "hit@3": sum(chance_hit(n_chunks, m, 3) for m in relevant) / len(relevant),
            "mrr": sum(chance_mrr(n_chunks, m, RETRIEVAL_DEPTH) for m in relevant) / len(relevant),
            "recall@3": min(3, n_chunks) / n_chunks,
            "recall@5": min(5, n_chunks) / n_chunks,
        }

    return {
        "chunk_count": n_chunks,
        "chunk_hash": chunk_fingerprint(chunks),
        "store_points": stats["total_documents"],
        "question_counts": {c: sum(1 for q in questions if q.category == c) for c in CATEGORIES},
        "calibration_count": len(calibration),
        "depth": RETRIEVAL_DEPTH,
        "fallbacks": fallbacks,
        "thresholds": thresholds,
        "results": results,
        "chance": chance,
    }


# --------------------------------------------------------------------------- baseline


def baseline_view(report: dict) -> dict:
    """The part of a report worth pinning: chunk count, chunk hash and the point values."""
    numbers = {
        method: {
            category: {m: round(cell[m]["value"], 4) for m in cell if m != "n"}
            for category, cell in per_category.items()
        }
        for method, per_category in report["results"].items()
    }
    return {
        "chunk_count": report["chunk_count"],
        "chunk_hash": report["chunk_hash"],
        "tolerance": BASELINE_TOLERANCE,
        "numbers": numbers,
    }


@dataclass
class Verdict:
    chunking: List[str]
    regressions: List[str]

    @property
    def ok(self) -> bool:
        return not self.chunking and not self.regressions


def compare_to_baseline(report: dict, baseline: dict) -> Verdict:
    """Two distinct outcomes. If the chunks changed, the numbers are not comparable, so only that is reported (a
    dependency bump of the text splitter must read as "chunking changed", never as a retrieval regression)."""
    chunking = []
    if report["chunk_count"] != baseline["chunk_count"]:
        chunking.append(f"chunk count is {report['chunk_count']}, the baseline has {baseline['chunk_count']}")
    if report["chunk_hash"] != baseline["chunk_hash"]:
        chunking.append("the hash of the chunk texts differs from the baseline")
    if chunking:
        return Verdict(chunking, [])
    tolerance = baseline["tolerance"]
    regressions = []
    current = baseline_view(report)["numbers"]
    for method, per_category in baseline["numbers"].items():
        for category, metrics in per_category.items():
            for metric, pinned in metrics.items():
                now = current.get(method, {}).get(category, {}).get(metric)
                if now is None:
                    regressions.append(f"{method}/{category}/{metric} is missing from this run")
                elif now < pinned - tolerance:
                    regressions.append(f"{method}/{category}/{metric} fell from {pinned} to {now} (tolerance {tolerance})")
    return Verdict([], regressions)


# --------------------------------------------------------------------------- printing


def _cell(entry: dict) -> str:
    return f"{entry['value']:.2f} ({entry['lo']:.2f}-{entry['hi']:.2f})"


def format_report(report: dict, title: str) -> str:
    out = [title, ""]
    counts = ", ".join(f"{c} {n}" for c, n in report["question_counts"].items())
    out.append(f"corpus: {report['chunk_count']} chunks (hash {report['chunk_hash'][:12]}), retrieval depth k={report['depth']}")
    out.append(f"questions: {sum(report['question_counts'].values())} ({counts}); intervals are 95% bootstrap over questions,")
    out.append(f"{BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}. With only {report['chunk_count']} chunks the intervals are wide.")
    out.append(f"hybrid fallbacks (hybrid_search answered with plain vector search): {report['fallbacks']}")
    out.append(
        f"abstain thresholds, calibrated on {report['calibration_count']} separate questions (calibration.jsonl): "
        + ", ".join(f"{m} {t:.4f}" for m, t in report["thresholds"].items())
    )
    out.append("")
    header = f"{'method':<7} {'category':<15} {'n':>3}  " + "  ".join(f"{m:<17}" for m in ANSWERABLE_METRICS)
    out.append(header)
    for method, per_category in report["results"].items():
        for category, cell in per_category.items():
            if category == "unanswerable":
                continue
            line = f"{method:<7} {category:<15} {cell['n']:>3}  " + "  ".join(f"{_cell(cell[m]):<17}" for m in ANSWERABLE_METRICS)
            out.append(line)
    out.append("")
    out.append("chance baseline (random ranking of the same chunks):")
    for category, cell in report["chance"].items():
        out.append(
            f"  {category:<15} " + "  ".join(f"{m} {cell[m]:.2f}" for m in ("hit@1", "hit@3", "mrr", "recall@3", "recall@5"))
        )
    out.append("")
    out.append("unanswerable questions correctly declined (top score below threshold), and answerable ones kept:")
    for method, per_category in report["results"].items():
        declined = per_category.get("unanswerable", {}).get("abstain")
        kept = [per_category[c]["kept"]["value"] * per_category[c]["n"] for c in per_category if c != "unanswerable"]
        total = sum(per_category[c]["n"] for c in per_category if c != "unanswerable")
        out.append(
            f"  {method:<7} declined {_cell(declined) if declined else 'n/a'}   kept {sum(kept) / total:.2f}"
        )
    return "\n".join(out)
