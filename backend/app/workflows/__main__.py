"""
    python -m app.workflows worker                       start the Hatchet worker (leave it running)
    python -m app.workflows run archive.zip [--timeout S]  submit a zip, wait, print the result table

Run both from the backend folder with HATCHET_CLIENT_TOKEN, HATCHET_CLIENT_TLS_STRATEGY and QDRANT_URL set (docs/ingestion.md).
"""
import argparse
import sys


def _print_table(status: dict) -> None:
    print(f"run {status['run_id']}  state {status['state']}  archive {status.get('archive')}")
    if status.get("error"):
        print(f"refused: {status['error']['code']}: {status['error']['reason']}")
    for item in status.get("files", []):
        reason = f"  {item['reason']}" if item["reason"] else ""
        print(f"  {item['status']:<18} {item['chunks']:>5}  {item['file']}{reason}")
    counts = status["counts"]
    print(
        f"ingested {counts['ingested']}  skipped_duplicate {counts['skipped_duplicate']}  failed {counts['failed']}"
        f"  in {status.get('seconds', '?')} s"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("worker")
    run = sub.add_parser("run")
    run.add_argument("zip")
    run.add_argument("--timeout", type=float, default=900.0)
    run.add_argument("--quiet", action="store_true", help="print the counts line only")
    args = parser.parse_args(argv)

    from app.workflows.hatchet_client import is_configured

    if not is_configured():
        print("HATCHET_CLIENT_TOKEN is not set. See docs/ingestion.md.")
        return 2
    from app.workflows import ingest
    from app.workflows.trigger import submit_archive, wait_for_run

    if args.command == "worker":
        ingest.build_worker().start()
        return 0
    store = ingest.run_store()
    run_id = submit_archive(args.zip, store)
    status = wait_for_run(store, run_id, args.timeout)
    if status is None:
        print(f"run {run_id} did not finish within {args.timeout} s; GET /api/ingest/{run_id} shows where it is")
        return 1
    if args.quiet:
        counts = status["counts"]
        print(f"{status['state']}: ingested {counts['ingested']}, skipped_duplicate {counts['skipped_duplicate']}, "
              f"failed {counts['failed']}, {status.get('seconds')} s")
    else:
        _print_table(status)
    return 0 if status["state"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
