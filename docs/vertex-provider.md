# Vertex AI Gemini provider

*A Gemini model on Vertex AI as one more tier in the generation service, built and tested against a stand-in server written from the Google pages saved in this repo.*

## What it does

`LLM_PROVIDER=vertex` with `VERTEX_CREDENTIALS_FILE` (a Google Cloud service account key; `VERTEX_CREDENTIALS_JSON` takes the same JSON inline) makes a Gemini model answer first. Ollama and Groq stay behind it as fallbacks, so a 403 or a blocked reply falls through to the next tier. `LLM_PROVIDER=auto`, the default, keeps the original order (Ollama, Groq) and adds Vertex AI last when a key is configured. Settings are listed in [reference.md](reference.md#environment-variables).

[`backend/app/services/vertex.py`](../backend/app/services/vertex.py):

- **Authentication.** Signs the OAuth 2.0 JWT bearer assertion with RS256 (claims `iss`, `scope`, `aud`, `iat`, `exp`, one hour), posts it to the token endpoint, and reuses the access token until 60 seconds before `expires_in`. A 401 from the model drops the token and mints one new one, once.
- **Requests.** `generateContent` and `streamGenerateContent` with `contents`, `systemInstruction`, `generationConfig` (temperature, `maxOutputTokens`, `stopSequences`) and `safetySettings` (one threshold applied to the four harm categories).
- **Replies.** The text of the first candidate, with thought summaries left out. A blocked prompt (`promptFeedback.blockReason`), a finish reason other than STOP or MAX_TOKENS (SAFETY, RECITATION, BLOCKLIST, PROHIBITED_CONTENT, SPII, OTHER and the rest), an empty candidate list, a stream that ends with no finish reason or before its closing bracket, and a chunk that is not a response object each raise a named error. MAX_TOKENS returns the truncated text with `finish_reason` set.
- **Retries.** 429, 500, 502, 503, 504 and a dropped connection, at most twice, with a delay of 1 second doubling each time, or the `Retry-After` value when that is longer (capped at 60 seconds). Any other 4xx raises at once with Google's status name and message.
- **LangChain.** `VertexGeminiLLM` is a LangChain LLM, so `prompt | llm | StrOutputParser()` works for `invoke` and `stream` as it does for the Groq tier.

## What it was built and tested against

The tests talk to [`backend/tests/vertex_standin.py`](../backend/tests/vertex_standin.py), a server I wrote from the pages saved in [vendor-contract/vertex-ai/](vendor-contract/vertex-ai/README.md). It:

- verifies each assertion: three base64url parts, header RS256 and JWT, the five claims, `exp` at most an hour after `iat`, and the signature against the key's public half;
- issues tokens with a lifetime and refuses an expired one with a 401;
- validates every request body against the documented fields (roles, text parts, `systemInstruction`, typed `generationConfig`, documented harm categories and thresholds) and answers 400 INVALID_ARGUMENT otherwise;
- answers in the documented response shapes, whole, or streamed as a JSON array sent in pieces or as `data:` lines;
- plays scripted faults: 429 with `Retry-After`, 5xx, 403, a blocked prompt, a blocked reply, an empty candidate list, a reply that is not JSON, a chunk that is not JSON, a stream with no finish reason and a stream cut off before its closing bracket.

The RSA key is generated for each test run. No test reads a key from disk, and none calls a Google address. The stand-in is the service the tests talk to, so a first run with a real key and project is also the first comparison of the saved pages with Google's live service.

## Tests

```bash
cd backend && pip install -r requirements-ci.txt
python -m pytest tests/test_vertex_provider.py     # 114 tests, about 10 seconds
python eval/vertex_mutation_check.py               # 39 mutants, each removes one guard
python eval/vertex_mutation_check.py --list        # what each mutant removes
```

The 114 tests cover a normal answer, a streamed answer in both framings, an expired token refreshed once, a token refused twice, a 429 then success, a 403, a blocked prompt, a blocked reply for six finish reasons, a malformed chunk in both framings, the wrong signing key, a dropped connection, the provider order setting and a seeded property test of the token cache (60 random worlds of clock jumps and failing mints; a returned token is never past its expiry, a fresh cached token is never replaced, a failed mint leaves the cache alone).

The mutation check copies `app/` to a temporary folder, removes or breaks one guard (a retry status, the `Retry-After` read, the token margin, the 401 refresh, the assertion audience, the signature hash, the grant type, a request field, the blocked-reply check, the finish-reason check, the closing-bracket check, the provider order) and runs the tests against the copy. A mutant counts only when a test fails.

Output of the run on 2026-10-30: [pytest](test-runs/vertex-provider-pytest-2026-10-30.txt) and [mutation check](test-runs/vertex-provider-mutation-2026-10-30.txt) (39 of 39 caught). CI runs both.
