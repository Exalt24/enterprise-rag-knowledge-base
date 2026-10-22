# Scenario 3: a question that needs two documents

## User request

> A change request adds a new risk and will be paid for from contingency. What has to happen in the risk register, and how does the drawdown show up in the cost report?

## A correct agent

1. Sees that the question has two parts that live in different documents, and calls `search_documents` once per part rather than once with the whole sentence.
2. First search, about the register: finds `risk-register-procedure.md#6`, which says a change request that adds or alters a risk must update the register before the change is approved.
3. Second search, about the cost report: finds `monthly-cost-report-guide.md#4`, which says each contingency drawdown appears as its own line with the CR number beside it.
4. Answers both parts and puts the matching citation id next to each claim, so a reader can see which document supports which half.

## Failures this scenario catches

- Searching once and answering only the half that the first results cover.
- Citing one document for both claims.
- Merging the two sources into one sentence that neither document states.
- Citing `risk-register-procedure.md` alone when the cost report claim came from the other file.

## Machine-checked

```json
{
  "id": "two-documents-two-searches",
  "skill_section": "How to choose a tool",
  "calls": [
    {
      "tool": "search_documents",
      "arguments": {"query": "change request that adds a risk must update the register before approval", "limit": 5},
      "outcome": "hits",
      "expect_ids": ["risk-register-procedure.md#6"]
    },
    {
      "tool": "search_documents",
      "arguments": {"query": "contingency drawdown shown as its own line in the cost report", "limit": 5},
      "outcome": "hits",
      "expect_ids": ["monthly-cost-report-guide.md#4"]
    }
  ],
  "cite": ["risk-register-procedure.md#6", "monthly-cost-report-guide.md#4"],
  "facts": ["must update the register before the change is approved", "appears as its own line"],
  "absent_terms": [],
  "extra_chunks": []
}
```
