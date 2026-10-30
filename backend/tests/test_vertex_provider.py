"""The Vertex AI provider against a stand-in server written from the saved Google pages (tests/vertex_standin.py).

Both directions are covered: what the provider SENDS is checked by the stand-in (a wrong assertion, a bad request body
or an expired token is refused there), and what the provider RECEIVES is scripted by the stand-in (a normal answer, a
streamed one, a 429, a 403, a blocked reply, a chunk that is not JSON). No network, no key: the RSA key is generated
here for each run.

    pytest tests/test_vertex_provider.py

eval/vertex_mutation_check.py deletes each guard in app/services/vertex.py one at a time and requires a test here to
fail for every one.
"""
import email.utils
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("RENDER", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
import requests  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from app.services import vertex  # noqa: E402
from app.services.vertex import (  # noqa: E402
    RetryPolicy,
    ServiceAccount,
    TokenProvider,
    VertexAPIError,
    VertexAuthError,
    VertexBlockedError,
    VertexClient,
    VertexError,
    VertexGeminiLLM,
    VertexResponseError,
    build_vertex_llm,
    normalize_finish_reason,
    parse_retry_after,
    sign_assertion,
)
from tests.vertex_standin import StandIn  # noqa: E402

EMAIL = "svc@demo-project.iam.gserviceaccount.com"


def pem_of(key) -> str:
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode("ascii")


@pytest.fixture(scope="module")
def key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def standin(key):
    server = StandIn(EMAIL, key.public_key()).start()
    yield server
    server.stop()


@pytest.fixture
def account(standin, key):
    return ServiceAccount(client_email=EMAIL, private_key=pem_of(key), token_uri=standin.token_uri, private_key_id="key-1", project_id="demo-project")


@pytest.fixture
def sleeps():
    return []


@pytest.fixture
def client(standin, account, sleeps):
    policy = RetryPolicy(sleep=sleeps.append)
    return VertexClient(account, project="demo-project", location="us-central1", model="gemini-3.5-flash", api_base=standin.base, policy=policy, timeout=10)


OPTIONS = dict(
    system_instruction="Answer only from the context.",
    generation_config={"temperature": 0.1, "maxOutputTokens": 256, "stopSequences": ["END"]},
    safety_settings=vertex.safety_settings_for("BLOCK_ONLY_HIGH"),
)


# ------------------------------------------------------------------ a normal answer

def test_a_normal_answer_is_mapped_and_the_request_has_the_documented_shape(client, standin):
    result = client.generate("How many vacation days?", **OPTIONS)
    assert result.text == standin.answer
    assert result.finish_reason == "STOP"
    assert result.usage["totalTokenCount"] > 0
    sent = standin.generate_requests[-1]["body"]
    assert sent["contents"] == [{"role": "user", "parts": [{"text": "How many vacation days?"}]}]
    assert sent["systemInstruction"] == {"parts": [{"text": "Answer only from the context."}]}
    assert sent["generationConfig"]["stopSequences"] == ["END"]
    assert {s["category"] for s in sent["safetySettings"]} == set(vertex.HARM_CATEGORIES)
    assert standin.generate_requests[-1]["problem"] is None
    assert standin.generate_requests[-1]["method"] == "generateContent"


def test_the_signed_assertion_is_accepted_by_the_token_endpoint(client, standin):
    client.generate("hello")
    form = standin.token_requests[0]
    assert form["grant_type"] == ["urn:ietf:params:oauth:grant-type:jwt-bearer"]
    assert len(standin.token_requests) == 1


def test_the_assertion_carries_the_documented_claims(account):
    import base64

    jwt = sign_assertion(account, vertex.CLOUD_PLATFORM_SCOPE, now=1_700_000_000.0)
    header, claims, _ = (json.loads(base64.urlsafe_b64decode(p + "=" * (-len(p) % 4))) for p in jwt.split(".")[:2] + ["e30"])
    assert header == {"alg": "RS256", "typ": "JWT", "kid": "key-1"}
    assert claims["iss"] == EMAIL
    assert claims["scope"] == "https://www.googleapis.com/auth/cloud-platform"
    assert claims["aud"] == account.token_uri
    assert claims["iat"] == 1_700_000_000 and claims["exp"] == claims["iat"] + 3600


def test_an_assertion_signed_with_the_wrong_key_is_refused_with_a_clear_error(standin, sleeps):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    bad = ServiceAccount(client_email=EMAIL, private_key=pem_of(other), token_uri=standin.token_uri)
    c = VertexClient(bad, "demo-project", "us-central1", "gemini-3.5-flash", api_base=standin.base, policy=RetryPolicy(sleep=sleeps.append))
    with pytest.raises(VertexAuthError, match="invalid_grant.*signature"):
        c.generate("hello")
    assert standin.generate_requests == []  # never got as far as the model
    assert sleeps == []  # a 400 from the token endpoint is not retried


def test_a_private_key_that_is_not_a_key_is_a_clear_error(standin):
    broken = ServiceAccount(client_email=EMAIL, private_key="not a pem", token_uri=standin.token_uri)
    with pytest.raises(VertexAuthError, match="could not be used to sign"):
        sign_assertion(broken, vertex.CLOUD_PLATFORM_SCOPE, 0.0)


# ------------------------------------------------------------------ streaming

@pytest.mark.parametrize("framing", ["array", "sse"])
def test_a_streamed_answer_arrives_in_pieces_and_joins_to_the_answer(client, standin, framing):
    standin.script({"kind": "ok", "format": framing})
    pieces = list(client.stream("How many vacation days?", **OPTIONS))
    assert len(pieces) > 3
    assert "".join(pieces) == standin.answer
    assert standin.generate_requests[-1]["method"] == "streamGenerateContent"
    assert standin.generate_requests[-1]["problem"] is None


def test_a_streamed_chunk_that_is_not_json_raises_after_the_good_chunks(client, standin):
    standin.script({"kind": "malformed"})
    got = []
    with pytest.raises(VertexResponseError, match="not valid JSON"):
        for piece in client.stream("hello"):
            got.append(piece)
    assert got == ["The "]  # the chunk before the damaged one was already delivered


def test_a_malformed_sse_chunk_raises(client, standin):
    standin.script({"kind": "malformed", "format": "sse"})
    with pytest.raises(VertexResponseError, match="not valid JSON"):
        list(client.stream("hello"))


def test_a_stream_cut_off_before_the_closing_bracket_raises(client, standin):
    standin.script({"kind": "short_array"})
    with pytest.raises(VertexResponseError):
        list(client.stream("hello"))


def test_a_stream_with_no_finish_reason_raises(client, standin):
    standin.script({"kind": "no_finish", "format": "sse"})
    with pytest.raises(VertexResponseError, match="without a finish reason"):
        list(client.stream("hello"))


def test_a_stream_blocked_midway_raises_after_the_text_that_arrived(client, standin):
    standin.script({"kind": "reply_blocked", "reason": "SAFETY"})
    got = []
    with pytest.raises(VertexBlockedError) as info:
        for piece in client.stream("hello"):
            got.append(piece)
    assert info.value.reason == "SAFETY" and info.value.where == "reply"
    assert got  # the earlier chunks were delivered


# ------------------------------------------------------------------ tokens

def test_the_token_is_cached_between_calls(client, standin):
    for _ in range(3):
        client.generate("hello")
    assert len(standin.token_requests) == 1


def test_an_expired_token_is_refreshed_once_and_the_call_succeeds(client, standin):
    client.generate("first")
    standin.expire_every_token()
    result = client.generate("second")
    assert result.text == standin.answer
    assert len(standin.token_requests) == 2  # one refresh, not a loop
    assert standin.generate_requests[-1]["token"] == "stand-in-token-2"


def test_a_token_that_keeps_being_refused_fails_after_one_refresh(client, standin):
    standin.script({"status": 401, "code": "UNAUTHENTICATED"}, {"status": 401, "code": "UNAUTHENTICATED"})
    with pytest.raises(VertexAPIError) as info:
        client.generate("hello")
    assert info.value.status == 401
    assert len(standin.token_requests) == 2


def test_the_token_is_replaced_shortly_before_it_expires_without_waiting_for_a_401(standin, account, sleeps):
    clock = [1_000.0]
    policy = RetryPolicy(sleep=sleeps.append, clock=lambda: clock[0])
    c = VertexClient(account, "demo-project", "us-central1", "gemini-3.5-flash", api_base=standin.base, policy=policy)
    c.generate("a")
    clock[0] += 3600 - 120  # 120 seconds left, outside the 60 second margin
    c.generate("b")
    assert len(standin.token_requests) == 1
    clock[0] += 70  # now 50 seconds left, inside the margin
    c.generate("c")
    assert len(standin.token_requests) == 2


# ------------------------------------------------------------------ retries and errors

def test_a_429_is_retried_after_the_retry_after_and_then_succeeds(client, standin, sleeps):
    standin.script({"status": 429, "code": "RESOURCE_EXHAUSTED", "retry_after": 7})
    assert client.generate("hello").text == standin.answer
    assert sleeps == [7.0]
    assert len(standin.generate_requests) == 2


def test_a_5xx_is_retried_with_a_doubling_delay(client, standin, sleeps):
    standin.script({"status": 503, "code": "UNAVAILABLE"}, {"status": 500, "code": "INTERNAL"})
    assert client.generate("hello").text == standin.answer
    assert sleeps == [1.0, 2.0]


def test_a_retry_that_keeps_failing_gives_up_after_two_retries(client, standin, sleeps):
    standin.script(*[{"status": 429, "code": "RESOURCE_EXHAUSTED", "message": "quota"}] * 5)
    with pytest.raises(VertexAPIError, match="429 RESOURCE_EXHAUSTED: quota") as info:
        client.generate("hello")
    assert info.value.status == 429
    assert len(standin.generate_requests) == 3
    assert sleeps == [1.0, 2.0]


def test_a_streaming_429_is_retried_before_any_chunk(client, standin, sleeps):
    standin.script({"status": 429, "code": "RESOURCE_EXHAUSTED"})
    assert "".join(client.stream("hello")) == standin.answer
    assert sleeps == [1.0]


def test_a_403_is_refused_at_once_with_the_reason_and_is_not_retried(client, standin, sleeps):
    standin.script({"status": 403, "code": "PERMISSION_DENIED", "message": "The caller does not have permission"})
    with pytest.raises(VertexAPIError) as info:
        client.generate("hello")
    assert info.value.status == 403
    assert "PERMISSION_DENIED" in str(info.value) and "does not have permission" in str(info.value)
    assert sleeps == []
    assert len(standin.generate_requests) == 1


def test_a_request_the_documented_shape_forbids_comes_back_as_a_400_error(client, standin, sleeps):
    with pytest.raises(VertexAPIError, match="400 INVALID_ARGUMENT.*safetySettings"):
        client.generate("hello", safety_settings=[{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_EVERYTHING"}])
    assert sleeps == []


def test_a_dropped_connection_is_retried_then_reported(account, sleeps):
    policy = RetryPolicy(sleep=sleeps.append)
    dead = "http://127.0.0.1:9"  # nothing listens on the discard port
    broken = ServiceAccount(client_email=EMAIL, private_key=account.private_key, token_uri=dead + "/token")
    c = VertexClient(broken, "demo-project", "us-central1", "gemini-3.5-flash", api_base=dead, policy=policy, timeout=1)
    with pytest.raises(VertexError, match="could not reach Vertex AI after 3 tries"):
        c.generate("hello")
    assert sleeps == [1.0, 2.0]


def test_retry_after_accepts_seconds_and_a_date():
    assert parse_retry_after("5", now=100.0) == 5.0
    when = email.utils.format_datetime(datetime.fromtimestamp(1_000_010, tz=timezone.utc), usegmt=True)
    assert parse_retry_after(when, now=1_000_000.0) == pytest.approx(10.0, abs=0.5)
    assert parse_retry_after("soon", now=0.0) is None
    assert parse_retry_after(None, now=0.0) is None
    assert RetryPolicy(max_delay=60).delay(0, "500") == 60  # an absurd Retry-After is capped


# ------------------------------------------------------------------ blocked and odd replies

def test_a_reply_blocked_by_safety_raises_a_blocked_error_with_the_reason(client, standin):
    standin.script({"kind": "reply_blocked", "reason": "SAFETY"})
    with pytest.raises(VertexBlockedError) as info:
        client.generate("hello")
    assert info.value.reason == "SAFETY" and info.value.where == "reply"
    assert "blocked the reply: SAFETY" in str(info.value)


@pytest.mark.parametrize("reason", ["RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "OTHER", "FINISH_REASON_SAFETY"])
def test_every_finish_reason_other_than_stop_is_not_an_answer(client, standin, reason):
    standin.script({"kind": "reply_blocked", "reason": reason})
    with pytest.raises(VertexBlockedError):
        client.generate("hello")


def test_a_blocked_prompt_raises_a_blocked_error_naming_the_prompt(client, standin):
    standin.script({"kind": "prompt_blocked"})
    with pytest.raises(VertexBlockedError, match="blocked the prompt: SAFETY"):
        client.generate("hello")


def test_max_tokens_is_a_truncated_answer_not_an_error(client, standin):
    standin.script({"kind": "max_tokens"})
    result = client.generate("hello")
    assert result.finish_reason == "MAX_TOKENS" and result.text == standin.answer


def test_a_reply_with_no_candidates_is_a_response_error(client, standin):
    standin.script({"kind": "no_candidates"})
    with pytest.raises(VertexResponseError, match="no candidates"):
        client.generate("hello")


def test_a_reply_that_is_not_json_is_a_response_error(client, standin):
    standin.script({"kind": "not_json"})
    with pytest.raises(VertexResponseError, match="did not return JSON"):
        client.generate("hello")


def test_finish_reason_names_are_normalised():
    assert normalize_finish_reason("FINISH_REASON_STOP") == "STOP"
    assert normalize_finish_reason("STOP") == "STOP"
    assert normalize_finish_reason("FINISH_REASON_UNSPECIFIED") == ""
    assert normalize_finish_reason(None) == ""


@pytest.mark.parametrize(
    "obj",
    [[], {"candidates": "x"}, {"candidates": ["x"]}, {"candidates": [{"content": {"parts": "x"}}]}, {"candidates": [{"content": {"parts": [{"text": 5}]}}]}],
)
def test_a_chunk_with_the_wrong_shape_is_a_response_error(obj):
    with pytest.raises(VertexResponseError):
        vertex.extract_chunk(obj)


def test_thought_parts_are_not_part_of_the_answer():
    text, finish = vertex.extract_chunk({"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True}, {"text": "42"}]}, "finishReason": "STOP"}]})
    assert (text, finish) == ("42", "STOP")


# ------------------------------------------------------------------ LangChain wrapper and the generation service

def test_the_wrapper_works_in_a_prompt_chain_for_invoke_and_stream(client, standin):
    llm = VertexGeminiLLM(client=client, system_instruction="Be brief.", safety_threshold="BLOCK_ONLY_HIGH")
    chain = ChatPromptTemplate.from_template("Context: {context}\nQuestion: {question}\nAnswer:") | llm | StrOutputParser()
    assert chain.invoke({"context": "25 days", "question": "days?"}) == standin.answer
    assert "".join(chain.stream({"context": "25 days", "question": "days?"})) == standin.answer
    first = standin.generate_requests[0]["body"]
    assert "Context: 25 days" in first["contents"][0]["parts"][0]["text"]
    assert first["systemInstruction"]["parts"][0]["text"] == "Be brief."
    assert first["generationConfig"]["temperature"] == 0.1


def test_stop_sequences_reach_the_request(client, standin):
    VertexGeminiLLM(client=client).invoke("hello", stop=["END"])
    assert standin.generate_requests[-1]["body"]["generationConfig"]["stopSequences"] == ["END"]


class _Settings:
    vertex_credentials_file = None
    vertex_credentials_json = None
    vertex_project = None
    vertex_location = "us-central1"
    vertex_model = "gemini-3.5-flash"
    vertex_api_base = None
    vertex_safety_threshold = None
    vertex_system_instruction = None
    request_timeout = 30


def test_the_provider_is_absent_without_a_service_account_and_builds_from_inline_json(standin, account):
    assert build_vertex_llm(_Settings()) is None
    s = _Settings()
    s.vertex_credentials_json = json.dumps({"client_email": EMAIL, "private_key": account.private_key, "token_uri": standin.token_uri, "project_id": "demo-project"})
    s.vertex_api_base = standin.base
    llm = build_vertex_llm(s)
    assert llm.invoke("hello") == standin.answer  # project came from the key, the call went through the stand-in


def test_a_service_account_file_with_no_key_is_a_clear_error(tmp_path):
    path = tmp_path / "sa.json"
    path.write_text(json.dumps({"client_email": EMAIL}), encoding="utf-8")
    with pytest.raises(VertexAuthError, match="private_key"):
        vertex.load_service_account(file_path=str(path))
    with pytest.raises(VertexAuthError, match="no service account configured"):
        vertex.load_service_account()


def test_the_service_account_repr_never_shows_the_key(account):
    assert "BEGIN" not in repr(account)


class _Llm:
    def __init__(self, name, fail=False):
        self.name, self.fail = name, fail


def _service(monkeypatch, provider, ollama=True, groq=True, vertex_on=True):
    from app.services import generation as gen

    monkeypatch.setattr(gen.settings, "llm_provider", provider)
    service = gen.GenerationService.__new__(gen.GenerationService)
    service.ollama = _Llm("o") if ollama else None
    service.groq = _Llm("g") if groq else None
    service.vertex = _Llm("v") if vertex_on else None
    return service


@pytest.mark.parametrize(
    "provider,expected",
    [("auto", ["ollama", "groq", "vertex"]), ("vertex", ["vertex", "ollama", "groq"]), ("groq", ["groq", "ollama", "vertex"]), ("VERTEX", ["vertex", "ollama", "groq"]), ("nonsense", ["ollama", "groq", "vertex"])],
)
def test_the_provider_setting_orders_the_tiers(monkeypatch, provider, expected):
    labels = [name.split("/")[0] for _, name in _service(monkeypatch, provider)._tiers()]
    assert labels == expected


def test_naming_a_provider_that_is_not_configured_falls_back_to_the_rest(monkeypatch):
    service = _service(monkeypatch, "vertex", vertex_on=False)
    assert [n.split("/")[0] for _, n in service._tiers()] == ["ollama", "groq"]


def test_the_generation_service_answers_from_vertex_when_it_is_first_and_falls_through_when_it_fails(monkeypatch, client, standin):
    service = _service(monkeypatch, "vertex", ollama=False, groq=False)
    service.vertex = VertexGeminiLLM(client=client)
    from langchain_core.prompts import ChatPromptTemplate as T

    service.prompt_template = T.from_template("{context} {question}")
    response = service.generate("days?", "25 days")
    assert response.model_used.startswith("vertex/") and response.answer == standin.answer
    assert service.primary_model_name.startswith("vertex/")

    model_used, stream = service.generate_stream("days?", "25 days")
    assert model_used.startswith("vertex/") and "".join(stream) == standin.answer

    standin.script({"status": 403, "code": "PERMISSION_DENIED"})
    down = service.generate("days?", "25 days")
    assert down.model_used == "none" and "unavailable" in down.answer  # the 403 surfaced as a failed tier, not a crash


# ------------------------------------------------------------------ property test: the token cache

class _FakeToken:
    def __init__(self, status, body):
        self.status_code, self._body, self.headers, self.text, self.reason = status, body, {}, json.dumps(body), ""

    def json(self):
        return self._body

    def close(self):
        pass


class _FakeSession:
    """Token endpoint stand-in for the property test: it answers from a script the seed produced."""

    def __init__(self, rng, clock):
        self.rng, self.clock = rng, clock
        self.issued = {}  # token -> expiry on the fake clock
        self.calls = 0

    def post(self, url, data=None, timeout=None, **_):
        self.calls += 1
        roll = self.rng.random()
        if roll < 0.15:
            return _FakeToken(400, {"error": "invalid_grant", "error_description": "scripted"})
        if roll < 0.25:
            raise requests.ConnectionError("scripted drop")
        lifetime = self.rng.choice([30, 90, 600, 3600, 7200])
        token = f"t{self.calls}"
        self.issued[token] = self.clock[0] + lifetime
        return _FakeToken(200, {"access_token": token, "expires_in": lifetime})


@pytest.mark.parametrize("seed", range(60))
def test_token_cache_invariants_over_random_worlds(seed, account, monkeypatch):
    """Whatever the clock does and whichever mints fail: a returned token is never past its expiry, a token is never
    minted while a cached one still has more than the margin left, a failure never leaves the cache poisoned, and
    a failed get is followed by a normal one."""
    monkeypatch.setattr(vertex, "sign_assertion", lambda *args, **kwargs: "unsigned.for.the.cache.test")  # RSA is not what is under test here
    rng = random.Random(seed)
    clock = [rng.uniform(1_000, 5_000)]
    session = _FakeSession(rng, clock)
    policy = RetryPolicy(max_retries=0, sleep=lambda s: None, clock=lambda: clock[0])
    tokens = TokenProvider(account, session=session, policy=policy, margin=60.0)
    last_good = None
    for _ in range(80):
        action = rng.random()
        if action < 0.5:
            clock[0] += rng.choice([0, 1, 5, 59, 61, 600, 3500, 4000])
        elif action < 0.6:
            tokens.invalidate()
            last_good = None
        calls_before, cached = session.calls, tokens._token
        cache_fresh = cached is not None and clock[0] < tokens._expires_at - 60.0
        try:
            token = tokens.get()
        except VertexError:
            assert tokens._token == cached  # a failed mint leaves the previous cache untouched
            assert session.calls - calls_before <= 1
            continue
        assert clock[0] < session.issued[token], "returned a token that has already expired"
        if cache_fresh:
            assert session.calls == calls_before and token == cached, "minted while a fresh token was cached"
        last_good = token
    assert last_good is None or last_good in session.issued
