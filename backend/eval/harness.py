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
from dataclasses import dataclass, field
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
LABEL_CATEGORIES = CATEGORIES + ("answerable",)  # calibration.jsonl does not split its answerable questions by style
METHODS = ("vector", "bm25", "hybrid")
ANSWERABLE_METRICS = ("hit@1", "hit@3", "mrr", "recall@3", "recall@5", "kept")
UNANSWERABLE_METRICS = ("abstain",)
# The README table: metric key and column title. A multi-document question counts as a hit@k when ANY one needed chunk is
# in the top k, while recall@k counts how many of the needed chunks are, so a table without recall hides how much was found.
TABLE_METRICS = (("hit@1", "hit@1"), ("hit@3", "hit@3"), ("mrr", "MRR"), ("recall@3", "recall@3"), ("recall@5", "recall@5"))
RETRIEVAL_DEPTH = 5  # chunks kept per query; recall@5 and MRR are measured inside this window
# The eval's own chunking, fixed here so a CHUNK_SIZE or CHUNK_OVERLAP in the environment (read by the app's settings)
# cannot move it. They equal the app defaults today; a change to either changes the chunk hash and reads as CHUNKING CHANGED.
EVAL_CHUNK_SIZE = 500
EVAL_CHUNK_OVERLAP = 50
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


class LabelError(ValueError):
    """A question file holds a label that cannot be scored. The message names every bad question by its id."""


def _read_rows(path: Path) -> Tuple[List[Tuple[int, object]], List[str]]:
    rows, problems = [], []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append((line_no, json.loads(line)))
        except json.JSONDecodeError as exc:
            problems.append(f"line {line_no}: not valid JSON ({exc.msg})")
    return rows, problems


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _need_problems(who: str, document, phrase, docs: Dict[str, str]) -> List[str]:
    if not _text(document):
        return [f"{who}: expected_document is empty or missing"]
    if document not in docs:
        return [f"{who}: expected_document {document!r} is not a file in the corpus ({', '.join(sorted(docs))})"]
    if not _text(phrase):
        return [f"{who}: must_contain is empty or missing for {document}"]
    if phrase not in docs[document]:
        elsewhere = [name for name, text in docs.items() if phrase in text]
        hint = f" (it is in {', '.join(elsewhere)})" if elsewhere else ""
        return [f"{who}: the phrase {phrase!r} is not in {document}{hint}"]
    return []


def label_problems(rows: Sequence[Tuple[int, object]], docs: Dict[str, str]) -> List[str]:
    """Every reason a row cannot be scored, each starting with the question id (or the line number when there is none)."""
    problems, seen = [], set()
    for line_no, row in rows:
        if not isinstance(row, dict):
            problems.append(f"line {line_no}: the row is not a JSON object")
            continue
        qid = row.get("id")
        who = qid if _text(qid) else f"line {line_no}"
        if not _text(qid):
            problems.append(f"{who}: id is empty or missing")
        elif qid in seen:
            problems.append(f"{who}: id is used twice")
        else:
            seen.add(qid)
        if not _text(row.get("question")):
            problems.append(f"{who}: question text is empty or missing")
        category = row.get("category")
        if category not in LABEL_CATEGORIES:
            problems.append(f"{who}: category {category!r} is not one of {', '.join(LABEL_CATEGORIES)}")
            continue
        extras = row.get("also", [])
        if category == "unanswerable":
            if row.get("expected_document") or row.get("must_contain") or extras:
                problems.append(f"{who}: an unanswerable question must have no expected_document, must_contain or also")
            continue
        problems += _need_problems(who, row.get("expected_document"), row.get("must_contain"), docs)
        if not isinstance(extras, list) or not all(isinstance(e, dict) for e in extras):
            problems.append(f"{who}: also must be a list of objects with expected_document and must_contain")
            continue
        for extra in extras:
            problems += _need_problems(who, extra.get("expected_document"), extra.get("must_contain"), docs)
        if category == "multi-document" and not extras:
            problems.append(f"{who}: a multi-document question needs at least one entry in also")
    return problems


def load_questions(path: Path = QUESTIONS_PATH, corpus_dir: Path = CORPUS_DIR, validate: bool = True) -> List[Question]:
    """Read a question file. With validate (the default) every label is checked against the corpus first and one
    LabelError names each bad question, so a wrong label never surfaces later as an error from deep inside the metrics."""
    rows, problems = _read_rows(path)
    if validate:
        docs = {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(corpus_dir).glob("*.md"))}
        problems += label_problems(rows, docs)
        if problems:
            raise LabelError(f"{Path(path).name} has {len(problems)} bad label(s): " + "; ".join(problems))
    questions = []
    for _, row in rows:
        needs = []
        if row.get("must_contain"):
            needs.append(Need(row.get("expected_document", ""), row["must_contain"]))
        for extra in row.get("also", []):
            needs.append(Need(extra.get("expected_document", ""), extra.get("must_contain", "")))
        questions.append(Question(row.get("id", ""), row.get("category", ""), row.get("question", ""), tuple(needs)))
    return questions


def check_labels_against_chunks(questions: Sequence[Question], chunks: Sequence[Document]) -> None:
    """A phrase can sit in its document and still be cut in half by the splitter; then no chunk can ever satisfy it."""
    problems = [
        f"{q.id}: no chunk of {n.document} holds the phrase {n.phrase!r} (a wrong document in the label, or the splitter cut the phrase)"
        for q in questions
        for n in q.needs
        if not any(chunk_satisfies(n, c) for c in chunks)
    ]
    if problems:
        raise LabelError(f"{len(problems)} bad label(s) against the chunks: " + "; ".join(problems))


def questions_fingerprint(questions: Sequence[Question]) -> str:
    """Hash of every question's id, category, text and needs: it changes when a question or a label changes, not when the
    file is only reformatted."""
    payload = json.dumps(
        [[q.id, q.category, q.text, [[n.document, n.phrase] for n in q.needs]] for q in questions], ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    chunks = TextChunker.chunk_documents(list(docs), chunk_size=EVAL_CHUNK_SIZE, chunk_overlap=EVAL_CHUNK_OVERLAP)
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

    @property
    def fallback_cause(self) -> str:
        """The first exception behind a fallback as one line ("RuntimeError: text"), or "" when there was none."""
        cause = self.engine.hybrid_fallback_cause
        return " ".join(str(cause).split())[:300] if cause else ""

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
    return percentile_bounds(means)


def percentile_bounds(sorted_means: Sequence[float]) -> Tuple[float, float]:
    """The 2.5th and 97.5th percentile of the resampled means: with N means, N/40 of them lie below the lower bound and
    N/40 above the upper one. When every resample lands on the same value (a cell where every question scored the same)
    both bounds equal it and the interval has no spread."""
    n = len(sorted_means)
    return sorted_means[int(0.025 * n)], sorted_means[int(0.975 * n) - 1]


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
    check_labels_against_chunks(list(questions) + list(calibration), chunks)
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
        fallback_cause = retrievers.fallback_cause

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
        "questions_hash": questions_fingerprint(questions),
        "calibration_hash": questions_fingerprint(calibration),
        "store_points": stats["total_documents"],
        "question_counts": {c: sum(1 for q in questions if q.category == c) for c in CATEGORIES},
        "calibration_count": len(calibration),
        "depth": RETRIEVAL_DEPTH,
        "fallbacks": fallbacks,
        "fallback_cause": fallback_cause,
        "fallback_methods": ["hybrid"] if fallbacks else [],  # only hybrid_search can fall back; its numbers are then vector numbers
        "thresholds": thresholds,
        "results": results,
        "chance": chance,
    }


# --------------------------------------------------------------------------- baseline


def report_numbers(report: dict) -> dict:
    """method -> category -> metric -> the point value rounded to 4 places (the form the baseline pins)."""
    return {
        method: {
            category: {m: round(cell[m]["value"], 4) for m in cell if m != "n"}
            for category, cell in per_category.items()
        }
        for method, per_category in report["results"].items()
    }


def baseline_view(report: dict) -> dict:
    """The part of a report worth pinning: chunk count, chunk hash, the hashes of the two question files, the point values."""
    return {
        "chunk_count": report["chunk_count"],
        "chunk_hash": report["chunk_hash"],
        "questions_hash": report["questions_hash"],
        "calibration_hash": report["calibration_hash"],
        "tolerance": BASELINE_TOLERANCE,
        "numbers": report_numbers(report),
    }


@dataclass
class Verdict:
    chunking: List[str]
    regressions: List[str]
    questions: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)  # methods whose numbers were not judged because they fell back

    @property
    def ok(self) -> bool:
        return not self.chunking and not self.regressions and not self.questions


def compare_to_baseline(report: dict, baseline: dict) -> Verdict:
    """Three distinct outcomes, and the first two stop the comparison because the numbers are then not comparable.

    The chunks changed (a dependency bump of the text splitter): CHUNKING CHANGED. A question or a label changed in
    questions.jsonl or calibration.jsonl: QUESTIONS CHANGED. Neither may ever read as a retrieval regression.
    Otherwise each pinned number is judged one-sided: it may rise freely and may fall by at most the tolerance. A method
    that fell back to vector-only is not judged at all, since its numbers are not its own; the fallback already fails.
    """
    chunking = []
    if report["chunk_count"] != baseline["chunk_count"]:
        chunking.append(f"chunk count is {report['chunk_count']}, the baseline has {baseline['chunk_count']}")
    if report["chunk_hash"] != baseline["chunk_hash"]:
        chunking.append("the hash of the chunk texts differs from the baseline")
    questions = []
    for key, name in (("questions_hash", "questions.jsonl"), ("calibration_hash", "calibration.jsonl")):
        if report.get(key) != baseline.get(key):
            questions.append(f"{name} differs from the one the baseline was pinned on, or the baseline does not record it")
    if chunking or questions:
        return Verdict(chunking, [], questions)
    tolerance = baseline["tolerance"]
    fell_back = set(report.get("fallback_methods", []))
    regressions = []
    current = report_numbers(report)
    for method, per_category in baseline["numbers"].items():
        if method in fell_back:
            continue
        for category, metrics in per_category.items():
            for metric, pinned in metrics.items():
                now = current.get(method, {}).get(category, {}).get(metric)
                if now is None:
                    regressions.append(f"{method}/{category}/{metric} is missing from this run")
                    continue
                fall = round(pinned - now, 4)
                if fall > tolerance:
                    regressions.append(
                        f"{method}/{category}/{metric} fell from {pinned} to {now}, by {fall:g}, "
                        f"more than the tolerance of {tolerance:g}"
                    )
    return Verdict([], regressions, [], sorted(fell_back & set(baseline["numbers"])))


def verdict_lines(verdict: Verdict) -> List[str]:
    """The text for each outcome, shared by `run_eval.py --check` and the baseline tests so they cannot drift apart."""
    repin = "re-pin with `python eval/run_eval.py --write-baseline`"
    lines = [f"CHUNKING CHANGED: {line}. This is not a retrieval regression. Read the new chunks, then {repin}." for line in verdict.chunking]
    lines += [
        f"QUESTIONS CHANGED: {line}. No number was compared, because a new label is not a retrieval regression. "
        f"Read the question diff, then {repin}."
        for line in verdict.questions
    ]
    lines += [f"REGRESSION: {line}" for line in verdict.regressions]
    lines += [f"NOT COMPARED: {m} fell back to vector-only, so its numbers are not its own." for m in verdict.skipped]
    return lines


# --------------------------------------------------------------------------- printing


def _cell(entry: dict) -> str:
    return f"{entry['value']:.2f} ({entry['lo']:.2f}-{entry['hi']:.2f})"


def format_report(report: dict, title: str) -> str:
    out = [title, ""]
    counts = ", ".join(f"{c} {n}" for c, n in report["question_counts"].items())
    out.append(f"corpus: {report['chunk_count']} chunks (hash {report['chunk_hash'][:12]}), retrieval depth k={report['depth']}")
    out.append(f"questions: {sum(report['question_counts'].values())} ({counts}); intervals are 95% bootstrap over questions,")
    out.append(f"{BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}. With only {report['chunk_count']} chunks the intervals are wide.")
    out.append("A cell at the ceiling prints a zero-width interval: every resample hits the same value, so there is no spread.")
    cause = f" (first cause: {report['fallback_cause']})" if report.get("fallback_cause") else ""
    out.append(f"hybrid fallbacks (hybrid_search answered with plain vector search): {report['fallbacks']}{cause}")
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


def markdown_table(report: dict, with_chance: bool) -> str:
    """The README results table, generated from a report so no number in it is typed by hand. Every row carries hit@1,
    hit@3, MRR, recall@3 and recall@5; with_chance adds the random-ranking rows."""
    header = ["method", "category", "n"] + [title for _, title in TABLE_METRICS]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for method, per_category in report["results"].items():
        for category, cell in per_category.items():
            if category == "unanswerable":
                continue
            values = [f"{cell[key]['value']:.2f} ({cell[key]['lo']:.2f} to {cell[key]['hi']:.2f})" for key, _ in TABLE_METRICS]
            out.append("| " + " | ".join([method, category, str(cell["n"])] + values) + " |")
    if with_chance:
        for category, cell in report["chance"].items():
            values = [f"{cell[key]:.2f}" for key, _ in TABLE_METRICS]
            out.append("| " + " | ".join(["chance", category, ""] + values) + " |")
    return "\n".join(out)
