# Week 1 Learning Progress Tracker

Track your progress through Week 1 learning objectives.

---

## Day 1-2: LangChain Fundamentals

### Part 1: Core Concepts (2 hours)
- [ ] Read LangChain Introduction
- [ ] Understand LCEL (LangChain Expression Language)
- [ ] Learn Prompt Templates
- [ ] Run `exercises/01_basics.py` successfully
- [ ] Can explain: What is LCEL?

**Notes:**
```
[Write your notes here - what clicked? What was confusing?]
```

### Part 2: Chains & Output Parsers (2 hours)
- [ ] Read about Chains
- [ ] Understand Output Parsers
- [ ] Learn Runnable Interface
- [ ] Complete `exercises/02_chains.py` (create this after 01_basics.py)
- [ ] Can explain: Why chain operations together?

**Notes:**
```
[Your notes]
```

### Part 3: Retrieval & Vector Stores (2 hours)
- [ ] Read about Embeddings
- [ ] Understand Vector Stores
- [ ] Learn Retrievers
- [ ] Complete `exercises/03_retrieval.py`
- [ ] Can explain: How does semantic search work?

**Notes:**
```
[Your notes]
```

---

## Day 3: RAG Architecture

### Part 1: RAG Fundamentals (1.5 hours)
- [ ] Read "What is RAG?" article
- [ ] Study RAG Architecture Patterns
- [ ] Learn Advanced RAG Techniques
- [ ] Can explain: Why use RAG instead of fine-tuning?

**Key Concepts Checklist:**
- [ ] Understand retrieval vs generation
- [ ] Know 3+ chunking strategies
- [ ] Understand embedding quality impact
- [ ] Know what context window means
- [ ] Understand source attribution

**Notes:**
```
[Your notes]
```

### Part 2: Build Complete RAG Chain (1.5 hours)
- [ ] Complete `exercises/04_rag_complete.py`
- [ ] Experiment with different k values (number of retrieved docs)
- [ ] Test queries both in and out of context
- [ ] Can explain: How does retrieval improve LLM answers?

**Observations:**
```
k=1 results: [note what happened]
k=3 results: [note what happened]
k=5 results: [note what happened]

Best k value for this use case: ___ because ___
```

---

## Day 4: Design Your Architecture

### Design Document
- [ ] Create `ARCHITECTURE.md`
- [ ] Draw system diagram (can be text-based)
- [ ] List all components with responsibilities
- [ ] Justify technology choices
- [ ] Define success metrics

**Architecture Decisions:**
```
Why Llama 3 over GPT-4?
[Your reasoning]

Why Chroma over Pinecone?
[Your reasoning]

Chunking strategy choice:
[Your reasoning]
```

---

## Checkpoint Exercises

### Exercise 1: Concept Explanation
Write 2-3 sentence explanations for:

- [ ] **What is LCEL?**
  ```
  [Your explanation]
  ```

- [ ] **Embeddings vs LLM - what's the difference?**
  ```
  [Your explanation]
  ```

- [ ] **Why chunk documents?**
  ```
  [Your explanation]
  ```

- [ ] **What makes a good RAG prompt?**
  ```
  [Your explanation]
  ```

### Exercise 2: Build from Scratch
- [ ] Complete `exercises/05_from_scratch.py`
- [ ] No looking at other files!
- [ ] Got it working? ✅

**Challenges faced:**
```
[What was hard? How did you solve it?]
```

### Exercise 3: Experimentation
- [ ] Test chunk size: 100 tokens
- [ ] Test chunk size: 500 tokens
- [ ] Test chunk size: 1000 tokens

**Results:**
```
100 tokens:
- Retrieval quality: ___/10
- Speed: ___

500 tokens:
- Retrieval quality: ___/10
- Speed: ___

1000 tokens:
- Retrieval quality: ___/10
- Speed: ___

Winner: ___ because ___
```

---

## Time Tracking

| Day | Task | Estimated | Actual | Notes |
|-----|------|-----------|--------|-------|
| 1 | LangChain Part 1 | 2h | | |
| 1 | LangChain Part 2 | 2h | | |
| 2 | LangChain Part 3 | 2h | | |
| 3 | RAG Fundamentals | 1.5h | | |
| 3 | Build RAG Chain | 1.5h | | |
| 4 | Design Architecture | 2-3h | | |
| **Total** | **10-12h** | | | |

---

## Week 1 Completion Criteria

Check ALL before moving to Week 2:

- [ ] Can build a RAG chain from scratch without tutorials
- [ ] Can explain why each component exists
- [ ] Have designed your system architecture
- [ ] Understand trade-offs (chunking, retrieval strategies)
- [ ] Completed all checkpoint exercises
- [ ] `ARCHITECTURE.md` created and reviewed

---

## Questions / Blockers

**Questions I still have:**
1.
2.
3.

**What I need to review:**
1.
2.
3.

**What clicked for me:**
1.
2.
3.

---

## Ready for Week 2?

If you checked all completion criteria above, you're ready to start building the production system!

**Week 2 Preview:**
- Document ingestion pipeline (PDF, DOCX, TXT)
- Chunking implementation
- Vector database setup
- Basic query system

**Next steps:**
1. Commit your learning progress to git
2. Review your architecture design one more time
3. Get ready to write production code!
