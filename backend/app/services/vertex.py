"""
Vertex AI Gemini provider.

Calls `generateContent` and `streamGenerateContent` on a Gemini model through the REST API, with a service
account for authentication, and presents the result to the generation service as a LangChain LLM, the same
shape the Ollama and Groq tiers have. The request and response shapes follow the pages saved in
docs/vendor-contract/vertex-ai/.

Pieces, from the bottom up:
- sign_assertion: the OAuth 2.0 JWT bearer assertion, RS256, claims iss, scope, aud, iat, exp (one hour).
- TokenProvider: exchanges the assertion at the token endpoint and caches the access token until shortly before
  it expires, so a request does not pay for a token round trip.
- send_with_retry: retries 429 and 5xx (and a dropped connection) with exponential backoff, waiting at least
  as long as a Retry-After header says. Any other 4xx is returned to the caller, never retried.
- VertexClient: builds the request, maps the response (and every streamed chunk) to text, and raises a typed
  error for a blocked prompt, a blocked reply, a finish reason other than STOP or MAX_TOKENS, a stream that
  ends without a finish reason, and a chunk that is not a response object.
- VertexGeminiLLM: the LangChain wrapper over the client.
"""

from __future__ import annotations

import base64
import codecs
import email.utils
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token"
JWT_BEARER_GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"

# Statuses worth another try: quota (429) and the server side hiccups the error page lists (500, 503), plus the
# gateway pair that sits in front of them.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
# A finish reason that ends an answer normally. Everything else (SAFETY, RECITATION, BLOCKLIST,
# PROHIBITED_CONTENT, SPII, OTHER, MALFORMED_FUNCTION_CALL, MODEL_ARMOR ...) means the reply is not a real answer.
GOOD_FINISH_REASONS = frozenset({"STOP", "MAX_TOKENS"})
HARM_CATEGORIES = (
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
)


class VertexError(Exception):
    """Base class: anything this provider raises."""


class VertexAuthError(VertexError):
    """The service account could not be turned into an access token."""


class VertexAPIError(VertexError):
    """A non-retriable error status from Vertex, or a retriable one that kept failing."""

    def __init__(self, status: int, message: str, code_name: str = ""):
        self.status = status
        self.code_name = code_name
        label = f" {code_name}" if code_name else ""
        super().__init__(f"Vertex AI returned HTTP {status}{label}: {message}")


class VertexBlockedError(VertexError):
    """The prompt or the reply was blocked, or the model stopped for a reason other than STOP or MAX_TOKENS."""

    def __init__(self, reason: str, where: str, detail: str = ""):
        self.reason = reason
        self.where = where
        suffix = f" ({detail})" if detail else ""
        super().__init__(f"Vertex AI blocked the {where}: {reason}{suffix}")


class VertexResponseError(VertexError):
    """A response or a streamed chunk that does not have the documented shape."""


# --------------------------------------------------------------------------------------
# Service account and the signed assertion
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ServiceAccount:
    client_email: str
    private_key: str
    token_uri: str = DEFAULT_TOKEN_URI
    private_key_id: str = ""
    project_id: str = ""

    def __repr__(self) -> str:  # never print the key
        return f"ServiceAccount(client_email={self.client_email!r}, private_key=<hidden>)"


def parse_service_account(info: Dict[str, Any]) -> ServiceAccount:
    for field_name in ("client_email", "private_key"):
        if not isinstance(info.get(field_name), str) or not info[field_name]:
            raise VertexAuthError(f"service account JSON has no usable '{field_name}'")
    return ServiceAccount(
        client_email=info["client_email"],
        private_key=info["private_key"],
        token_uri=info.get("token_uri") or DEFAULT_TOKEN_URI,
        private_key_id=info.get("private_key_id") or "",
        project_id=info.get("project_id") or "",
    )


def load_service_account(file_path: Optional[str] = None, json_text: Optional[str] = None) -> ServiceAccount:
    """Read a service account key from a file path or from inline JSON (an env var in a container)."""
    if json_text:
        raw = json_text
    elif file_path:
        with open(file_path, "r", encoding="utf-8") as handle:
            raw = handle.read()
    else:
        raise VertexAuthError("no service account configured (set VERTEX_CREDENTIALS_FILE or VERTEX_CREDENTIALS_JSON)")
    try:
        info = json.loads(raw)
    except ValueError as exc:
        raise VertexAuthError(f"service account JSON is not valid JSON: {exc}") from exc
    if not isinstance(info, dict):
        raise VertexAuthError("service account JSON is not an object")
    return parse_service_account(info)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def sign_assertion(account: ServiceAccount, scope: str, now: float) -> str:
    """Build and sign the JWT the token endpoint exchanges for an access token (RS256, valid for one hour)."""
    header: Dict[str, Any] = {"alg": "RS256", "typ": "JWT"}
    if account.private_key_id:
        header["kid"] = account.private_key_id
    issued = int(now)
    claims = {
        "iss": account.client_email,
        "scope": scope,
        "aud": account.token_uri,
        "iat": issued,
        "exp": issued + 3600,
    }
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        + "."
        + _b64url(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
    )
    try:
        key = serialization.load_pem_private_key(account.private_key.encode("utf-8"), password=None)
        signature = key.sign(signing_input.encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
    except (ValueError, TypeError) as exc:
        raise VertexAuthError(f"the service account private key could not be used to sign: {exc}") from exc
    return signing_input + "." + _b64url(signature)


# --------------------------------------------------------------------------------------
# Retry
# --------------------------------------------------------------------------------------

def parse_retry_after(value: Optional[str], now: float) -> Optional[float]:
    """Retry-After is a number of seconds or an HTTP date. Returns seconds to wait, or None if absent or unreadable."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
        return max(0.0, seconds)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - now)


@dataclass
class RetryPolicy:
    # The error page recommends at most two retries, starting at one second and doubling.
    max_retries: int = 2
    base_delay: float = 1.0
    max_delay: float = 60.0
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.time

    def delay(self, attempt: int, retry_after: Optional[str]) -> float:
        backoff = self.base_delay * (2 ** attempt)
        asked = parse_retry_after(retry_after, self.clock())
        wait = backoff if asked is None else max(backoff, asked)
        return min(wait, self.max_delay)


def send_with_retry(send: Callable[[], requests.Response], policy: RetryPolicy) -> requests.Response:
    """Call send() until it returns something that is not a retriable failure or the retries run out.

    The last response is returned either way, so the caller maps the final status to an error.
    """
    attempt = 0
    while True:
        try:
            response = send()
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt >= policy.max_retries:
                raise VertexError(f"could not reach Vertex AI after {attempt + 1} tries: {exc}") from exc
            policy.sleep(policy.delay(attempt, None))
            attempt += 1
            continue
        if response.status_code in RETRY_STATUSES and attempt < policy.max_retries:
            wait = policy.delay(attempt, response.headers.get("Retry-After"))
            response.close()
            policy.sleep(wait)
            attempt += 1
            continue
        return response


def _error_from(response: requests.Response) -> VertexAPIError:
    """Turn an error response into one readable exception, using Google's {"error": {...}} body when there is one."""
    message = ""
    code_name = ""
    try:
        body = response.json()
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            message = str(error.get("message") or "")
            code_name = str(error.get("status") or "")
        elif isinstance(body, dict) and body.get("error_description"):
            message = str(body["error_description"])
            code_name = str(body.get("error") or "")
    except ValueError:
        pass
    if not message:
        message = (response.text or "").strip()[:300] or response.reason or "no detail"
    return VertexAPIError(response.status_code, message, code_name)


# --------------------------------------------------------------------------------------
# Access token
# --------------------------------------------------------------------------------------

class TokenProvider:
    """Mints access tokens from a service account and reuses one until `margin` seconds before it expires."""

    def __init__(
        self,
        account: ServiceAccount,
        session: Optional[requests.Session] = None,
        policy: Optional[RetryPolicy] = None,
        scope: str = CLOUD_PLATFORM_SCOPE,
        margin: float = 60.0,
        timeout: float = 30.0,
    ):
        self.account = account
        self.session = session or requests.Session()
        self.policy = policy or RetryPolicy()
        self.scope = scope
        self.margin = margin
        self.timeout = timeout
        self._token: Optional[str] = None
        self._expires_at = 0.0
        self._lock = threading.Lock()
        self.mint_count = 0

    def invalidate(self) -> None:
        with self._lock:
            self._token = None
            self._expires_at = 0.0

    def get(self) -> str:
        with self._lock:
            now = self.policy.clock()
            if self._token and now < self._expires_at - self.margin:
                return self._token
            self._token, lifetime = self._mint(now)
            self._expires_at = now + lifetime
            return self._token

    def _mint(self, now: float):
        assertion = sign_assertion(self.account, self.scope, now)
        response = send_with_retry(
            lambda: self.session.post(
                self.account.token_uri,
                data={"grant_type": JWT_BEARER_GRANT, "assertion": assertion},
                timeout=self.timeout,
            ),
            self.policy,
        )
        if response.status_code != 200:
            error = _error_from(response)
            raise VertexAuthError(f"the token endpoint refused the service account assertion: {error}") from error
        try:
            body = response.json()
        except ValueError as exc:
            raise VertexAuthError("the token endpoint did not return JSON") from exc
        token = body.get("access_token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not token:
            raise VertexAuthError("the token endpoint reply has no access_token")
        try:
            lifetime = float(body.get("expires_in", 3600))
        except (TypeError, ValueError):
            lifetime = 3600.0
        self.mint_count += 1
        return token, lifetime


# --------------------------------------------------------------------------------------
# Response mapping
# --------------------------------------------------------------------------------------

@dataclass
class VertexResult:
    text: str
    finish_reason: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    model_version: str = ""


def normalize_finish_reason(value: Any) -> str:
    """'STOP' and 'FINISH_REASON_STOP' are the same thing, and an empty or unspecified reason means 'not finished'."""
    if not isinstance(value, str) or not value:
        return ""
    name = value[len("FINISH_REASON_"):] if value.startswith("FINISH_REASON_") else value
    return "" if name == "UNSPECIFIED" else name


def _check_prompt_feedback(obj: Dict[str, Any]) -> None:
    feedback = obj.get("promptFeedback")
    if not isinstance(feedback, dict):
        return
    reason = feedback.get("blockReason")
    if isinstance(reason, str) and reason and reason != "BLOCKED_REASON_UNSPECIFIED":
        raise VertexBlockedError(reason, "prompt", str(feedback.get("blockReasonMessage") or ""))


def extract_chunk(obj: Any) -> "tuple[str, str]":
    """One GenerateContentResponse (a whole reply or one streamed chunk) to (text, finish_reason).

    Raises VertexBlockedError for a blocked prompt, VertexResponseError when the object is not shaped like a response.
    """
    if not isinstance(obj, dict):
        raise VertexResponseError(f"expected a response object, got {type(obj).__name__}")
    _check_prompt_feedback(obj)
    candidates = obj.get("candidates")
    if candidates is None:
        return "", ""
    if not isinstance(candidates, list):
        raise VertexResponseError("'candidates' is not a list")
    if not candidates:
        return "", ""
    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise VertexResponseError("a candidate is not an object")
    content = candidate.get("content") or {}
    if not isinstance(content, dict):
        raise VertexResponseError("candidate 'content' is not an object")
    parts = content.get("parts") or []
    if not isinstance(parts, list):
        raise VertexResponseError("candidate 'parts' is not a list")
    pieces: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            raise VertexResponseError("a content part is not an object")
        text = part.get("text")
        if text is not None and not isinstance(text, str):
            raise VertexResponseError("a text part is not a string")
        if text and not part.get("thought"):  # thought summaries are not part of the answer
            pieces.append(text)
    finish = normalize_finish_reason(candidate.get("finishReason"))
    if finish and finish not in GOOD_FINISH_REASONS:
        raise VertexBlockedError(finish, "reply", str(candidate.get("finishMessage") or ""))
    return "".join(pieces), finish


def iter_stream_objects(response: requests.Response) -> Iterator[Any]:
    """Yield each JSON object of a streamed reply as it arrives.

    The body is a JSON array of response objects, sent a few at a time; `data:` lines are accepted too.
    A piece that does not parse raises VertexResponseError.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    sse = "text/event-stream" in (response.headers.get("Content-Type") or "")
    buffer = ""
    started = False
    finished = False
    json_decoder = json.JSONDecoder()

    def drain(final: bool) -> Iterator[Any]:
        nonlocal buffer, started, finished
        if sse:
            buffer = buffer.replace("\r\n", "\n")  # a lone \r at the end waits for its \n in the next piece
            while True:
                # a complete event ends in a blank line; at the end of the body the last one may not
                split = buffer.find("\n\n")
                if split < 0 and not (final and buffer.strip()):
                    return
                event, buffer = (buffer, "") if split < 0 else (buffer[:split], buffer[split + 2:])
                data = "\n".join(
                    line[5:].lstrip() for line in event.split("\n") if line.startswith("data:")
                )
                if data.strip() and data.strip() != "[DONE]":
                    try:
                        yield json.loads(data)
                    except ValueError as exc:
                        raise VertexResponseError(f"a streamed chunk is not valid JSON: {exc}") from exc
            return
        while True:
            buffer = buffer.lstrip(" \t\r\n,")
            if not buffer:
                return
            if not started:
                if buffer[0] != "[":
                    raise VertexResponseError("the stream does not start with a JSON array")
                buffer = buffer[1:]
                started = True
                continue
            if buffer[0] == "]":
                finished = True
                buffer = ""
                return
            try:
                obj, end = json_decoder.raw_decode(buffer)
            except ValueError as exc:
                if final:
                    raise VertexResponseError(f"a streamed chunk is not valid JSON: {exc}") from exc
                return  # probably just not all here yet; wait for more bytes
            buffer = buffer[end:]
            yield obj

    for raw in response.iter_content(chunk_size=None):
        if not raw:
            continue
        buffer += decoder.decode(raw)
        yield from drain(final=False)
    buffer += decoder.decode(b"", final=True)
    yield from drain(final=True)
    if not sse and started and not finished:
        raise VertexResponseError("the stream ended before the closing bracket")


# --------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------

def default_api_base(location: str) -> str:
    return "https://aiplatform.googleapis.com" if location == "global" else f"https://{location}-aiplatform.googleapis.com"


def safety_settings_for(threshold: Optional[str]) -> Optional[List[Dict[str, str]]]:
    """One threshold for all four harm categories, or None to leave the model's own defaults."""
    if not threshold:
        return None
    return [{"category": category, "threshold": threshold} for category in HARM_CATEGORIES]


class VertexClient:
    def __init__(
        self,
        account: ServiceAccount,
        project: str,
        location: str,
        model: str,
        api_base: Optional[str] = None,
        session: Optional[requests.Session] = None,
        policy: Optional[RetryPolicy] = None,
        timeout: float = 60.0,
        tokens: Optional[TokenProvider] = None,
    ):
        if not project:
            raise VertexError("no Google Cloud project configured (set VERTEX_PROJECT)")
        self.project = project
        self.location = location
        self.model = model
        self.api_base = (api_base or default_api_base(location)).rstrip("/")
        self.session = session or requests.Session()
        self.policy = policy or RetryPolicy()
        self.timeout = timeout
        self.tokens = tokens or TokenProvider(account, session=self.session, policy=self.policy)

    def _url(self, method: str) -> str:
        return (
            f"{self.api_base}/v1/projects/{self.project}/locations/{self.location}"
            f"/publishers/google/models/{self.model}:{method}"
        )

    def build_request(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        safety_settings: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if generation_config:
            body["generationConfig"] = generation_config
        if safety_settings:
            body["safetySettings"] = safety_settings
        return body

    def _post(self, method: str, body: Dict[str, Any], stream: bool) -> requests.Response:
        url = self._url(method)
        refreshed = False
        while True:
            token = self.tokens.get()
            response = send_with_retry(
                lambda: self.session.post(
                    url,
                    json=body,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=self.timeout,
                    stream=stream,
                ),
                self.policy,
            )
            if response.status_code == 401 and not refreshed:
                # the cached token was rejected (expired early, or revoked): mint one new token and try once more
                refreshed = True
                response.close()
                self.tokens.invalidate()
                continue
            if response.status_code != 200:
                error = _error_from(response)
                response.close()
                raise error
            return response

    def generate(self, prompt: str, **options: Any) -> VertexResult:
        response = self._post("generateContent", self.build_request(prompt, **options), stream=False)
        try:
            obj = response.json()
        except ValueError as exc:
            raise VertexResponseError("generateContent did not return JSON") from exc
        text, finish = extract_chunk(obj)
        if not isinstance(obj, dict) or not obj.get("candidates"):
            raise VertexResponseError("generateContent returned no candidates")
        usage = obj.get("usageMetadata")
        return VertexResult(
            text=text,
            finish_reason=finish,
            usage=usage if isinstance(usage, dict) else {},
            model_version=str(obj.get("modelVersion") or ""),
        )

    def stream(self, prompt: str, **options: Any) -> Iterator[str]:
        """Yield the text of each streamed chunk. Raises after the last good chunk if the stream ends badly."""
        response = self._post("streamGenerateContent", self.build_request(prompt, **options), stream=True)
        finish = ""
        try:
            for obj in iter_stream_objects(response):
                text, chunk_finish = extract_chunk(obj)
                if chunk_finish:
                    finish = chunk_finish
                if text:
                    yield text
        finally:
            response.close()
        if not finish:
            raise VertexResponseError("the stream ended without a finish reason, so the answer may be cut short")


# --------------------------------------------------------------------------------------
# LangChain wrapper
# --------------------------------------------------------------------------------------

class VertexGeminiLLM(LLM):
    """A Gemini model on Vertex AI as a LangChain LLM, so `prompt | llm | StrOutputParser()` works as it does for Groq."""

    client: Any
    system_instruction: Optional[str] = None
    temperature: float = 0.1
    max_output_tokens: Optional[int] = None
    safety_threshold: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "vertex-gemini"

    def _options(self, stop: Optional[List[str]]) -> Dict[str, Any]:
        config: Dict[str, Any] = {"temperature": self.temperature}
        if self.max_output_tokens:
            config["maxOutputTokens"] = self.max_output_tokens
        if stop:
            config["stopSequences"] = list(stop)
        return {
            "system_instruction": self.system_instruction,
            "generation_config": config,
            "safety_settings": safety_settings_for(self.safety_threshold),
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self.client.generate(prompt, **self._options(stop)).text

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for delta in self.client.stream(prompt, **self._options(stop)):
            chunk = GenerationChunk(text=delta)
            if run_manager:
                run_manager.on_llm_new_token(delta, chunk=chunk)
            yield chunk


def build_vertex_llm(settings: Any) -> Optional[VertexGeminiLLM]:
    """The configured provider, or None when no service account is set (so the tier is simply absent)."""
    if not (settings.vertex_credentials_file or settings.vertex_credentials_json):
        return None
    account = load_service_account(settings.vertex_credentials_file, settings.vertex_credentials_json)
    project = settings.vertex_project or account.project_id
    client = VertexClient(
        account,
        project=project,
        location=settings.vertex_location,
        model=settings.vertex_model,
        api_base=settings.vertex_api_base or None,
        timeout=float(settings.request_timeout),
    )
    return VertexGeminiLLM(
        client=client,
        system_instruction=settings.vertex_system_instruction or None,
        safety_threshold=settings.vertex_safety_threshold or None,
    )
