"""Mutation check for the Vertex AI provider: delete or break one guard at a time and require a test to fail.

Each mutant is an exact piece of source text and its replacement, applied to a COPY of app/ and the Vertex tests in a
temporary directory (the working tree is never touched). The mutant is CAUGHT only when pytest exits with code 1, which
means a test failed. Exit code 0 means the mutant SURVIVED (no test noticed the guard was gone); any other code means the
copy did not run (a collection error) and counts as a failure of the check, not as proof of anything. A mutant whose
source text is not found exactly once also fails the check: the code moved and this list is stale.

    python eval/vertex_mutation_check.py            run every mutant (four at a time)
    python eval/vertex_mutation_check.py --list     list them
    python eval/vertex_mutation_check.py --only retry-after-ignored
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List

BACKEND = Path(__file__).resolve().parents[1]
VERTEX = "app/services/vertex.py"
GENERATION = "app/services/generation.py"
TESTS = ["tests/test_vertex_provider.py"]


@dataclass(frozen=True)
class Mutant:
    name: str
    path: str
    old: str
    new: str
    guard: str  # what the removed line protects, in a few words


MUTANTS: List[Mutant] = [
    # retries
    Mutant("retry-429-removed", VERTEX, "frozenset({429, 500, 502, 503, 504})", "frozenset({500, 502, 503, 504})", "a 429 is retried"),
    Mutant("retry-5xx-removed", VERTEX, "frozenset({429, 500, 502, 503, 504})", "frozenset({429})", "a 5xx is retried"),
    Mutant("retry-403-added", VERTEX, "frozenset({429, 500, 502, 503, 504})", "frozenset({403, 429, 500, 502, 503, 504})", "a 4xx other than 429 is never retried"),
    Mutant("retry-after-ignored", VERTEX, "wait = backoff if asked is None else max(backoff, asked)", "wait = backoff", "Retry-After is honoured"),
    Mutant("retry-after-header-not-read", VERTEX, 'response.headers.get("Retry-After"))', "None)", "the Retry-After header is passed to the delay"),
    Mutant("backoff-flat", VERTEX, "self.base_delay * (2 ** attempt)", "self.base_delay", "the delay doubles"),
    Mutant("delay-cap-removed", VERTEX, "return min(wait, self.max_delay)", "return wait", "an absurd Retry-After is capped"),
    Mutant("one-retry-too-many", VERTEX, "if response.status_code in RETRY_STATUSES and attempt < policy.max_retries:", "if response.status_code in RETRY_STATUSES and attempt < policy.max_retries + 1:", "retries stop at the limit"),
    Mutant("connection-error-not-retried", VERTEX, "except (requests.ConnectionError, requests.Timeout) as exc:", "except ZeroDivisionError as exc:", "a dropped connection is retried then reported"),
    # errors
    Mutant("error-status-name-dropped", VERTEX, 'code_name = str(error.get("status") or "")', 'code_name = ""', "the 4xx error names Google's status"),
    Mutant("non-200-accepted", VERTEX, "            if response.status_code != 200:\n                error = _error_from(response)\n                response.close()", "            if False:\n                error = _error_from(response)\n                response.close()", "a 4xx from the model raises"),
    Mutant("token-refusal-swallowed", VERTEX, "        if response.status_code != 200:\n            error = _error_from(response)\n            raise VertexAuthError", "        if False:\n            error = _error_from(response)\n            raise VertexAuthError", "a refused assertion raises an auth error"),
    # token cache and refresh
    Mutant("token-cache-disabled", VERTEX, "if self._token and now < self._expires_at - self.margin:", "if False:", "the token is reused until shortly before expiry"),
    Mutant("token-margin-removed", VERTEX, "now < self._expires_at - self.margin", "now < self._expires_at", "the token is replaced before it expires"),
    Mutant("token-lifetime-ignored", VERTEX, "self._expires_at = now + lifetime", "self._expires_at = now + 3600", "the cache uses expires_in from the reply"),
    Mutant("failed-mint-clears-cache", VERTEX, "            self._token, lifetime = self._mint(now)", "            self._token = None\n            self._token, lifetime = self._mint(now)", "a failed mint leaves the old cache alone"),
    Mutant("refresh-on-401-removed", VERTEX, "if response.status_code == 401 and not refreshed:", "if False:", "an expired token is refreshed"),
    Mutant("refresh-not-bounded", VERTEX, "                refreshed = True\n", "                refreshed = False\n", "the refresh happens once"),
    Mutant("stale-token-kept-after-401", VERTEX, "                self.tokens.invalidate()\n", "                pass\n", "the refused token is dropped"),
    # assertion and token request
    Mutant("assertion-wrong-audience", VERTEX, '"aud": account.token_uri,', '"aud": "https://example.invalid/token",', "aud names the token endpoint"),
    Mutant("assertion-lifetime-too-long", VERTEX, '"exp": issued + 3600,', '"exp": issued + 7200,', "exp is at most one hour after iat"),
    Mutant("assertion-wrong-hash", VERTEX, "padding.PKCS1v15(), hashes.SHA256())", "padding.PKCS1v15(), hashes.SHA512())", "the signature is RSA SHA-256"),
    Mutant("wrong-grant-type", VERTEX, 'JWT_BEARER_GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"', 'JWT_BEARER_GRANT = "urn:ietf:params:oauth:grant-type:jwt_bearer"', "the JWT bearer grant type"),
    Mutant("bearer-header-removed", VERTEX, 'headers={"Authorization": f"Bearer {token}"},', "headers={},", "every call carries the token"),
    # request shape
    Mutant("system-instruction-dropped", VERTEX, 'body["systemInstruction"] = {"parts": [{"text": system_instruction}]}', "pass", "systemInstruction is sent"),
    Mutant("generation-config-dropped", VERTEX, 'body["generationConfig"] = generation_config', "pass", "generationConfig is sent"),
    Mutant("safety-settings-dropped", VERTEX, 'body["safetySettings"] = safety_settings', "pass", "safetySettings are sent"),
    Mutant("stop-sequences-dropped", VERTEX, 'config["stopSequences"] = list(stop)', "pass", "stop sequences reach the request"),
    Mutant("stream-uses-the-wrong-method", VERTEX, 'self._post("streamGenerateContent"', 'self._post("generateContent"', "streaming calls streamGenerateContent"),
    # reply mapping
    Mutant("prompt-block-ignored", VERTEX, 'if isinstance(reason, str) and reason and reason != "BLOCKED_REASON_UNSPECIFIED":', "if False:", "a blocked prompt raises"),
    Mutant("blocked-finish-reason-accepted", VERTEX, "if finish and finish not in GOOD_FINISH_REASONS:", "if False:", "a finish reason other than STOP raises"),
    Mutant("max-tokens-treated-as-error", VERTEX, 'GOOD_FINISH_REASONS = frozenset({"STOP", "MAX_TOKENS"})', 'GOOD_FINISH_REASONS = frozenset({"STOP"})', "a truncated answer is still an answer"),
    Mutant("finish-prefix-not-stripped", VERTEX, 'name = value[len("FINISH_REASON_"):] if value.startswith("FINISH_REASON_") else value', "name = value", "FINISH_REASON_ names are normalised"),
    Mutant("no-candidates-accepted", VERTEX, 'if not isinstance(obj, dict) or not obj.get("candidates"):', "if False:", "an empty reply raises"),
    Mutant("thought-parts-included", VERTEX, 'if text and not part.get("thought"):', "if text:", "thought summaries stay out of the answer"),
    # streaming
    Mutant("malformed-chunk-swallowed", VERTEX, "                if final:\n                    raise VertexResponseError", "                if False:\n                    raise VertexResponseError", "a chunk that is not JSON raises"),
    Mutant("stream-without-finish-accepted", VERTEX, "        if not finish:\n", "        if False:\n", "a stream with no finish reason raises"),
    Mutant("unclosed-array-accepted", VERTEX, "if not sse and started and not finished:", "if False:", "a stream cut off before the closing bracket raises"),
    # the provider setting
    Mutant("provider-setting-ignored", GENERATION, "            order.insert(0, wanted)", "            order.append(wanted)", "the provider setting moves a tier to the front"),
]


def apply(mutant: Mutant, root: Path) -> None:
    target = root / mutant.path
    text = target.read_text(encoding="utf-8")
    count = text.count(mutant.old)
    if count != 1:
        raise LookupError(f"{mutant.name}: expected its source text exactly once in {mutant.path}, found {count}")
    target.write_text(text.replace(mutant.old, mutant.new), encoding="utf-8")


def run_one(mutant: Mutant) -> dict:
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="vertex-mutant-") as tmp:
        root = Path(tmp)
        shutil.copytree(BACKEND / "app", root / "app", ignore=shutil.ignore_patterns("__pycache__"))
        (root / "tests").mkdir()
        for name in ("vertex_standin.py", "test_vertex_provider.py"):
            shutil.copy(BACKEND / "tests" / name, root / "tests" / name)
        shutil.copy(BACKEND / "pytest.ini", root / "pytest.ini")
        try:
            apply(mutant, root)
        except LookupError as exc:
            return {"mutant": mutant, "status": "STALE", "detail": str(exc), "seconds": 0.0}
        env = dict(os.environ, RENDER="true", PYTHONDONTWRITEBYTECODE="1")
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *TESTS],
            cwd=root, env=env, capture_output=True, text=True, timeout=400,
        )
        failed = [line for line in done.stdout.splitlines() if line.startswith("FAILED")]
        status = {1: "CAUGHT", 0: "SURVIVED"}.get(done.returncode, "BROKEN")
        detail = failed[0].split(" - ")[0].replace("FAILED ", "") if failed else (done.stdout + done.stderr)[-300:]
        return {"mutant": mutant, "status": status, "detail": detail, "seconds": time.time() - started}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    chosen = [m for m in MUTANTS if not args.only or m.name in args.only]
    if args.only and len(chosen) != len(set(args.only)):
        print("unknown mutant name in --only")
        return 2
    if args.list:
        for m in chosen:
            print(f"{m.name:34} {m.guard}")
        return 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(run_one, chosen))
    bad = 0
    for r in results:
        m = r["mutant"]
        print(f"{r['status']:9} {m.name:34} {m.guard:58} -> {r['detail']}")
        bad += r["status"] != "CAUGHT"
    caught = sum(r["status"] == "CAUGHT" for r in results)
    print(f"\n{caught} of {len(results)} mutants caught, {bad} not")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
