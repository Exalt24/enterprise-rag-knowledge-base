# LIFESPAN: permanent
"""A stand-in for the two Google services the Vertex provider talks to, written from the pages saved in
docs/vendor-contract/vertex-ai/. It is NOT Google: it checks what the saved documentation says a caller must send
and answers in the documented shapes, so a test can break the provider in either direction (a bad request is
refused here, a bad reply is produced here) without a network or a key.

Token endpoint (POST /token):
  - form body with grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer and an assertion
  - the assertion is verified: three base64url parts, header alg RS256 and typ JWT, claims iss, scope, aud, iat, exp with
    exp - iat at most 3600, and an RSA SHA-256 signature that checks against the service account's PUBLIC key
  - reply {"access_token", "token_type": "Bearer", "expires_in"}; a wrong assertion gets 400 {"error": "invalid_grant"}

Generate endpoints (POST /v1/projects/{p}/locations/{l}/publishers/google/models/{m}:generateContent and
:streamGenerateContent):
  - Authorization: Bearer must be a token this server issued and has not expired, else 401 UNAUTHENTICATED
  - the body is validated against the documented request: contents (role user or model, parts with text), an optional
    systemInstruction with parts, generationConfig with typed fields, safetySettings with documented categories and
    thresholds; anything else is 400 INVALID_ARGUMENT
  - the reply is a GenerateContentResponse; the streamed form is a JSON array of them (or data: lines), sent in pieces

Faults are queued per generate call with `script(...)`; each call consumes the next one, and an empty queue means a
normal answer.
"""

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

CATEGORIES = {
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
}
THRESHOLDS = {"OFF", "BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH"}
GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"


def _unb64(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def _error(code: int, status: str, message: str) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message, "status": status}}


class StandIn:
    def __init__(self, client_email: str, public_key, project: str = "demo-project", location: str = "us-central1", model: str = "gemini-3.5-flash"):
        self.client_email = client_email
        self.public_key = public_key
        self.project, self.location, self.model = project, location, model
        self.answer = "The vacation policy gives 25 days a year."
        self.token_lifetime = 3600
        self.faults: List[Dict[str, Any]] = []
        self.token_requests: List[Dict[str, Any]] = []
        self.generate_requests: List[Dict[str, Any]] = []
        self.tokens: Dict[str, float] = {}  # token -> expiry in server time
        self.lock = threading.Lock()
        self.issued = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    # -- lifecycle ----------------------------------------------------------------

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    @property
    def token_uri(self) -> str:
        return self.base + "/token"

    def start(self) -> "StandIn":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def script(self, *faults: Dict[str, Any]) -> None:
        with self.lock:
            self.faults = list(faults)

    def expire_every_token(self) -> None:
        with self.lock:
            for token in self.tokens:
                self.tokens[token] = 0.0

    # -- internals ----------------------------------------------------------------

    def _next_fault(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.faults.pop(0) if self.faults else None

    def _verify_assertion(self, assertion: str) -> Optional[str]:
        """None when the assertion is acceptable, otherwise the reason it is not."""
        parts = assertion.split(".")
        if len(parts) != 3:
            return "assertion is not three parts"
        try:
            header = json.loads(_unb64(parts[0]))
            claims = json.loads(_unb64(parts[1]))
            signature = _unb64(parts[2])
        except ValueError:
            return "assertion is not base64url JSON"
        if header.get("alg") != "RS256" or header.get("typ") != "JWT":
            return "header must be RS256 and JWT"
        for name in ("iss", "scope", "aud", "iat", "exp"):
            if name not in claims:
                return f"claim {name} is missing"
        if claims["iss"] != self.client_email:
            return "iss is not the service account"
        if claims["aud"] != self.token_uri:
            return "aud is not the token endpoint"
        if not claims["scope"]:
            return "scope is empty"
        if not (0 < claims["exp"] - claims["iat"] <= 3600):
            return "exp must be after iat and at most one hour later"
        try:
            self.public_key.verify(signature, (parts[0] + "." + parts[1]).encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
        except Exception:
            return "signature does not verify against the service account key"
        return None

    def _validate_request(self, body: Any) -> Optional[str]:
        if not isinstance(body, dict):
            return "body must be an object"
        contents = body.get("contents")
        if not isinstance(contents, list) or not contents:
            return "contents is required"
        for item in contents:
            if not isinstance(item, dict) or item.get("role", "user") not in ("user", "model"):
                return "each content needs role user or model"
            parts = item.get("parts")
            if not isinstance(parts, list) or not parts or not all(isinstance(p, dict) and isinstance(p.get("text"), str) for p in parts):
                return "each content needs text parts"
        system = body.get("systemInstruction")
        if system is not None:
            if not isinstance(system, dict) or not isinstance(system.get("parts"), list) or not all(isinstance(p.get("text"), str) for p in system["parts"]):
                return "systemInstruction needs text parts"
        config = body.get("generationConfig")
        if config is not None:
            if not isinstance(config, dict):
                return "generationConfig must be an object"
            for name in ("temperature", "topP"):
                if name in config and not isinstance(config[name], (int, float)):
                    return f"{name} must be a number"
            for name in ("maxOutputTokens", "topK", "candidateCount"):
                if name in config and (not isinstance(config[name], int) or isinstance(config[name], bool)):
                    return f"{name} must be an integer"
            if "stopSequences" in config and not (isinstance(config["stopSequences"], list) and all(isinstance(s, str) for s in config["stopSequences"])):
                return "stopSequences must be a list of strings"
        settings = body.get("safetySettings")
        if settings is not None:
            if not isinstance(settings, list):
                return "safetySettings must be a list"
            for entry in settings:
                if entry.get("category") not in CATEGORIES or entry.get("threshold") not in THRESHOLDS:
                    return "safetySettings needs a documented category and threshold"
        extra = set(body) - {"contents", "systemInstruction", "generationConfig", "safetySettings", "tools", "toolConfig", "labels", "cachedContent"}
        if extra:
            return f"unknown field {sorted(extra)[0]}"
        return None

    def _answer_chunks(self, prompt: str) -> List[Dict[str, Any]]:
        words = self.answer.split(" ")
        pieces = [w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words)]
        chunks = [{"candidates": [{"content": {"role": "model", "parts": [{"text": piece}]}}], "modelVersion": self.model} for piece in pieces]
        chunks[-1]["candidates"][0]["finishReason"] = "STOP"
        chunks[-1]["usageMetadata"] = {"promptTokenCount": max(1, len(prompt) // 4), "candidatesTokenCount": len(words), "totalTokenCount": max(1, len(prompt) // 4) + len(words)}
        return chunks

    def _whole_answer(self, prompt: str) -> Dict[str, Any]:
        reply = {
            "candidates": [{"content": {"role": "model", "parts": [{"text": self.answer}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": max(1, len(prompt) // 4), "candidatesTokenCount": len(self.answer.split()), "totalTokenCount": max(1, len(prompt) // 4) + len(self.answer.split())},
            "modelVersion": self.model,
        }
        return reply

    def _handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"  # the body of a streamed reply ends when the connection closes

            def log_message(self, *args):  # keep test output quiet
                return

            def _send(self, status: int, payload: Any, headers: Optional[Dict[str, str]] = None):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                if self.path == "/token":
                    return self._token(raw)
                prefix = f"/v1/projects/{outer.project}/locations/{outer.location}/publishers/google/models/{outer.model}:"
                if self.path.startswith(prefix) and self.path[len(prefix):] in ("generateContent", "streamGenerateContent"):
                    return self._generate(raw, self.path[len(prefix):])
                self._send(404, _error(404, "NOT_FOUND", f"no such method {self.path}"))

            def _token(self, raw: bytes):
                form = parse_qs(raw.decode("utf-8"))
                outer.token_requests.append(form)
                if form.get("grant_type") != [GRANT] or len(form.get("assertion", [])) != 1:
                    return self._send(400, {"error": "unsupported_grant_type", "error_description": "grant_type must be the JWT bearer grant"})
                problem = outer._verify_assertion(form["assertion"][0])
                if problem:
                    return self._send(400, {"error": "invalid_grant", "error_description": problem})
                with outer.lock:
                    outer.issued += 1
                    token = f"stand-in-token-{outer.issued}"
                    outer.tokens[token] = time.time() + outer.token_lifetime
                self._send(200, {"access_token": token, "token_type": "Bearer", "expires_in": outer.token_lifetime})

            def _generate(self, raw: bytes, method: str):
                auth = self.headers.get("Authorization", "")
                token = auth[7:] if auth.startswith("Bearer ") else ""
                with outer.lock:
                    expiry = outer.tokens.get(token)
                if expiry is None or expiry <= time.time():
                    return self._send(401, _error(401, "UNAUTHENTICATED", "Request had invalid authentication credentials."))
                try:
                    body = json.loads(raw)
                except ValueError:
                    return self._send(400, _error(400, "INVALID_ARGUMENT", "body is not JSON"))
                problem = outer._validate_request(body)
                outer.generate_requests.append({"method": method, "body": body, "token": token, "problem": problem})
                if problem:
                    return self._send(400, _error(400, "INVALID_ARGUMENT", problem))
                prompt = body["contents"][-1]["parts"][0]["text"]
                fault = outer._next_fault() or {}
                kind = fault.get("kind", "ok")
                if "status" in fault:
                    headers = {"Retry-After": str(fault["retry_after"])} if "retry_after" in fault else {}
                    return self._send(fault["status"], _error(fault["status"], fault.get("code", "ERROR"), fault.get("message", "scripted failure")), headers)
                if kind == "prompt_blocked":
                    return self._send(200, {"promptFeedback": {"blockReason": "SAFETY", "blockReasonMessage": "the prompt was blocked"}, "usageMetadata": {"promptTokenCount": 3, "totalTokenCount": 3}} if method == "generateContent" else [{"promptFeedback": {"blockReason": "SAFETY"}}])
                if method == "generateContent":
                    if kind == "reply_blocked":
                        return self._send(200, {"candidates": [{"finishReason": fault.get("reason", "SAFETY"), "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH", "blocked": True}]}]})
                    if kind == "no_candidates":
                        return self._send(200, {"candidates": []})
                    if kind == "not_json":
                        data = b"<html>gateway page</html>"
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        return
                    if kind == "max_tokens":
                        reply = outer._whole_answer(prompt)
                        reply["candidates"][0]["finishReason"] = "MAX_TOKENS"
                        return self._send(200, reply)
                    return self._send(200, outer._whole_answer(prompt))
                self._stream(prompt, fault)

            def _stream(self, prompt: str, fault: Dict[str, Any]):
                kind = fault.get("kind", "ok")
                chunks = outer._answer_chunks(prompt)
                sse = fault.get("format") == "sse"
                if kind == "reply_blocked":
                    chunks = chunks[:2]
                    chunks.append({"candidates": [{"finishReason": fault.get("reason", "SAFETY")}]})
                elif kind == "no_finish":
                    chunks = chunks[:-1]
                    chunks[-1]["candidates"][0].pop("finishReason", None)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream" if sse else "application/json")
                self.end_headers()
                write = lambda text: (self.wfile.write(text.encode("utf-8")), self.wfile.flush())
                if sse:
                    for index, chunk in enumerate(chunks):
                        text = json.dumps(chunk)
                        if kind == "malformed" and index == 1:
                            text = text[: len(text) // 2]
                        write("data: " + text + "\r\n\r\n")
                    return
                write("[")
                for index, chunk in enumerate(chunks):
                    text = json.dumps(chunk)
                    if kind == "malformed" and index == 1:
                        text = text[: len(text) // 2] + "}}}"  # a chunk that is not JSON
                    if kind == "short_array" and index == len(chunks) - 1:
                        write(("," if index else "") + text)
                        return  # the closing bracket never arrives
                    middle = len(text) // 2
                    write(("," if index else "") + text[:middle])  # the object arrives in two pieces
                    time.sleep(0.005)
                    write(text[middle:] + "\n")
                write("]")

        return Handler
