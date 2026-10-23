# Error messages and what to do

Each row is a message the knowledge base tools return as an error result, and the action to take. The messages are written by the server in `backend/mcp_server/server.py`.

The client shows the text with a prefix that names the tool, so a real error reads `Error executing tool search_documents: query is empty. Give a non-empty query of at most 500 characters.` Match a row against the text after that prefix.

| message after the prefix starts with | meaning | do this |
|---|---|---|
| `query is empty` or `question is empty` | The text was blank. | Build the query from the user's request and call again. |
| `query is N characters long; the limit is 500` | The text is too long. | Shorten it to the key terms. |
| `limit is N; it must be between 1 and 20` | `limit` is out of range. | Use a value from 1 to 20. |
| `Unknown document 'x'` | `document` is not an indexed file name. The message lists the real ones. | Pick the right name from that list, or call `list_documents`. |
| `Answer generation is unavailable` | No language model answered `ask_question`. | Use `search_documents` and answer from the snippets. Tell the user. |
| `The knowledge base is not reachable` | The vector store cannot be reached. The message names the setting to check. | Tell the user what the message says. Do not retry in a loop and do not answer from memory. |
| `The knowledge base failed` | Any other backend error, with its type and text. | Report the message once. Retry once only if it looks temporary. |
| `The knowledge base holds no documents yet` | The index is empty. This is a normal result, not an error. | Tell the user nothing is indexed. These tools cannot add documents. |
