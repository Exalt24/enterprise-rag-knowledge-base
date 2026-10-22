# Scenario 2: the documents do not cover the question

## User request

> What is the hotel allowance per night for business travel?

## A correct agent

1. Calls `search_documents` with the question's key terms. It does not skip the search because the topic sounds unrelated.
2. Reads the snippets. Search always returns its best matches, even when none of them answers the question, so a result list is not an answer. Here no snippet mentions hotels, travel or an allowance.
3. Tells the user that the indexed documents do not say, names what it searched for, and offers to try other wording or to list the documents with `list_documents`. It gives no figure and no citation.

## Failures this scenario catches

- Answering from general knowledge ("typically around ...") after a search that found nothing.
- Citing a chunk that was returned but does not mention the topic, to make the answer look sourced.
- Reading a high rank or a score as proof the chunk is relevant.
- Saying the documents say there is no allowance. They say nothing, which is different.

## Machine-checked

The test runs the search on the demo index and checks that none of the returned snippets contains any of the absent terms, so "nothing relevant" is the right reading of that result.

```json
{
  "id": "nothing-relevant-is-found",
  "skill_section": "When nothing is found",
  "calls": [
    {
      "tool": "search_documents",
      "arguments": {"query": "hotel allowance travel per night", "limit": 5},
      "outcome": "no_relevant_hits",
      "expect_ids": []
    }
  ],
  "cite": [],
  "facts": [],
  "absent_terms": ["hotel", "travel", "allowance"],
  "extra_chunks": []
}
```
