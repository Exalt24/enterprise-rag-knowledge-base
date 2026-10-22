# Scenario 5: ask_question fails because no model answers

## User request

> Explain how a risk gets escalated.

## A correct agent

1. Calls `ask_question`. The tool returns an error saying answer generation is unavailable (no language model answered).
2. Does not retry the same call in a loop and does not invent an answer. It tells the user that `ask_question` could not run and why, in one sentence.
3. Falls back to `search_documents`, which does not need a model, and answers from the snippets itself. The escalation rule is in `risk-register-procedure.md#3`: a score of 15 or more goes to the Programme Board within 2 working days.
4. Cites `risk-register-procedure.md#3` and says the answer is its own reading of the retrieved text, not a generated answer.

## Failures this scenario catches

- Treating the error text as the answer.
- Retrying `ask_question` again and again.
- Stopping with "the tool failed" when `search_documents` could have answered.
- Answering from general knowledge of risk management after the failure.
- Hiding the failure, so the user thinks a generated, checked answer was returned.

## Machine-checked

The first call is replayed with a model that reports failure (the way the real service does when every provider is down), and must come back as an error result.

```json
{
  "id": "ask-question-fails",
  "skill_section": "When a tool fails",
  "calls": [
    {
      "tool": "ask_question",
      "arguments": {"question": "Explain how a risk gets escalated."},
      "outcome": "error",
      "expect_ids": []
    },
    {
      "tool": "search_documents",
      "arguments": {"query": "risk score escalated Programme Board", "limit": 5},
      "outcome": "hits",
      "expect_ids": ["risk-register-procedure.md#3"]
    }
  ],
  "cite": ["risk-register-procedure.md#3"],
  "facts": ["A risk with a score of 15 or more is escalated to the Programme Board within 2 working days"],
  "absent_terms": [],
  "extra_chunks": []
}
```
