# Scenario 4: the user names a document

## User request

> In the risk register procedure, how often are critical risks reviewed?

## A correct agent

1. Does not guess a file name from the title the user spoke. Calls `list_documents` and finds `risk-register-procedure.md`.
2. Calls `search_documents` with `document` set to that exact file name, so chunks from other documents cannot crowd the answer out.
3. Reads the snippets, finds the sentence about the review cadence in `risk-register-procedure.md#2`, and answers that critical risks are reviewed every week.
4. Cites `risk-register-procedure.md#2`.

## Failures this scenario catches

- Passing a made-up file name as `document` and getting an error, then giving up.
- Searching all documents and citing a chunk from another document that happens to mention reviews.
- Answering with the right fact but citing a neighbouring chunk (`#3` or `#4`) that does not contain it.

## Machine-checked

```json
{
  "id": "user-names-one-document",
  "skill_section": "How to choose a tool",
  "calls": [
    {
      "tool": "list_documents",
      "arguments": {},
      "outcome": "documents",
      "expect_ids": []
    },
    {
      "tool": "search_documents",
      "arguments": {"query": "how often are critical risks reviewed", "limit": 5, "document": "risk-register-procedure.md"},
      "outcome": "hits",
      "expect_ids": ["risk-register-procedure.md#2"]
    }
  ],
  "cite": ["risk-register-procedure.md#2"],
  "facts": ["Critical risks are reviewed every week"],
  "absent_terms": [],
  "extra_chunks": []
}
```
