# Scenario 1: a direct question answered with ask_question and cited

## User request

> Who has to approve a change request that adds $100,000 to a work package?

## A correct agent

1. Calls `ask_question` once with the user's question. The question names one thing to look up, so a written answer with citations is enough.
2. Reads the answer and the returned `citations`. Checks that a cited snippet actually says who approves a change of that size (Tier 2, the Change Board).
3. Replies with the answer in its own words, followed by the citation id of the chunk that supports it: `change-control-procedure.md#1`.

## Failures this scenario catches

- Answering without any citation id.
- Citing an id that `ask_question` did not return, or one copied from memory of an earlier session.
- Passing on the model's answer without reading the citations, when the snippets do not say what the answer says.
- Answering from general knowledge of change control instead of calling the tool.

## Machine-checked

The test replays the calls on the demo index and checks that each expected id comes back, that every fact is in a cited chunk, and that every tool, argument and id is real. It does not run an agent.

```json
{
  "id": "approval-tier-with-ask-question",
  "skill_section": "How to choose a tool",
  "calls": [
    {
      "tool": "ask_question",
      "arguments": {"question": "Who has to approve a change request that adds $100,000 to a work package?"},
      "outcome": "answer",
      "expect_ids": ["change-control-procedure.md#1"]
    }
  ],
  "cite": ["change-control-procedure.md#1"],
  "facts": ["Tier 2 covers a change from $25,001 to $250,000", "The Change Board approves it"],
  "absent_terms": [],
  "extra_chunks": []
}
```
