"""Print the retrieval evaluation, or check it against the pinned baseline.

    python eval/run_eval.py                       deterministic table (hashing embedder), a HARNESS CHECK
    python eval/run_eval.py --check               same, exit 1 on a fallback, a chunking change or a regression
    python eval/run_eval.py --write-baseline      re-pin eval/baseline.json (only after reading why it changed)
    python eval/run_eval.py --embedder real       the repo's own embedding model, run locally, for a second table

The deterministic table proves the harness and the ranking code run; it uses a word-hashing embedder with no
semantics, so no quality claim and no "hybrid beats vector" claim may be drawn from it.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `eval` and `app` import when run as a file

from eval.embedders import HashingEmbeddings, real_embeddings  # noqa: E402
from eval.harness import (  # noqa: E402
    BASELINE_PATH,
    CALIBRATION_PATH,
    QUESTIONS_PATH,
    baseline_view,
    chunk_corpus,
    compare_to_baseline,
    format_report,
    load_corpus,
    load_questions,
    quiet,
    run_evaluation,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedder", choices=("hashing", "real"), default="hashing")
    parser.add_argument("--check", action="store_true", help="compare with baseline.json and fail on a problem")
    parser.add_argument("--write-baseline", action="store_true", help="write baseline.json from this run (hashing only)")
    parser.add_argument("--json", metavar="PATH", help="also write the full report as JSON")
    args = parser.parse_args(argv)

    if args.embedder == "real" and (args.check or args.write_baseline):
        print("The baseline pins the deterministic run only; use --embedder hashing with --check or --write-baseline.")
        return 2

    chunks = chunk_corpus(load_corpus())
    questions = load_questions(QUESTIONS_PATH)
    calibration = load_questions(CALIBRATION_PATH)

    if args.embedder == "hashing":
        embeddings = HashingEmbeddings()
        title = "DETERMINISTIC RUN (hashing embedder): HARNESS CHECK, not a quality measurement"
    else:
        with quiet():
            embeddings, backend = real_embeddings()
        title = f"REAL-MODEL RUN (repo default embedding model, backend {backend}): local run on a corpus of synthetic documents"

    report = run_evaluation(embeddings, questions, calibration, chunks)
    print(format_report(report, title))

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    status = 0
    if report["fallbacks"] > 0:
        print(f"\nFAIL: hybrid_search fell back to vector-only {report['fallbacks']} time(s), so the hybrid numbers are not hybrid.")
        status = 1

    if args.write_baseline:
        if status:
            print("Not writing the baseline from a run with fallbacks.")
            return status
        BASELINE_PATH.write_text(json.dumps(baseline_view(report), indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {BASELINE_PATH}")

    if args.check:
        verdict = compare_to_baseline(report, json.loads(BASELINE_PATH.read_text(encoding="utf-8")))
        for line in verdict.chunking:
            print(f"\nCHUNKING CHANGED: {line}. Read the new chunks, then re-pin with --write-baseline.")
        for line in verdict.regressions:
            print(f"\nREGRESSION: {line}")
        if verdict.ok and not status:
            print("\nbaseline check: OK")
        status = status or (0 if verdict.ok else 1)
    return status


if __name__ == "__main__":
    sys.exit(main())
