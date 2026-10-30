# Vertex AI vendor pages

*The Google pages the `vertex` provider was written against, saved as text on 2026-10-30 from a real Chromium load.*

The stand-in server in `backend/tests/vertex_standin.py` and the provider in `backend/app/services/vertex.py` follow these files, not memory. Each file starts with its URL.

| File | What the provider takes from it |
|---|---|
| `gemini-generate-content-request-and-response.txt` | request body (`contents`, `systemInstruction`, `generationConfig`, `safetySettings`), the four harm categories and thresholds, the response body, the finish reason list |
| `rest-models-generateContent.txt` | the endpoint shape `POST /v1/{model}:generateContent` and the model path `projects/{project}/locations/{location}/publishers/*/models/*` |
| `rest-models-streamGenerateContent.txt` | `POST /v1/{model}:streamGenerateContent`, whose response is a stream of `GenerateContentResponse` objects |
| `rest-GenerateContentResponse.txt` | `candidates[].finishReason`, `promptFeedback.blockReason`, `usageMetadata`, the `BlockedReason` and `FinishReason` enums |
| `oauth2-service-account-http-rest.txt` | the signed assertion (RS256, claims `iss`, `scope`, `aud`, `exp`, `iat`, at most one hour), the form POST with `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer`, the `expires_in` of the reply |
| `api-errors.txt` | 400, 401, 403, 404, 429, 500 and 503 meanings, and the advice to retry at most twice with a delay of at least one second that grows exponentially |

## What the pages do not say

These choices come from the provider's own design, so they are named here and are not claims about the saved pages.

- The OAuth scope the provider asks for is `https://www.googleapis.com/auth/cloud-platform`. The saved OAuth page lists no scope for this API.
- The service endpoint host is `{location}-aiplatform.googleapis.com` (`aiplatform.googleapis.com` for `global`). The REST pages say only `{service-endpoint}`.
- The stream is read as either a JSON array of response objects or `data:` lines, because the saved pages say "a stream of GenerateContentResponse instances" and do not fix the framing.
- Retry-After on a 429 or 5xx is honoured when present. The error page does not mention the header.
