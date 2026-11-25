# Benchmark: extractive

### 1. LLM used (model & parameters)
| Metric | Value |
|---|---|
| Backend / model | extractive / rule-based sentence ranker (no LLM) |
| Model parameters | 0 M |
| Model size in memory | 0 MB (n/a) on cpu |
| Context window (max tokens) | n/a |
| Embedding model | all-MiniLM-L6-v2, 22.7 M params, 384-dim |

### 2. Pipeline hyper-parameters
| Metric | Value |
|---|---|
| Knowledge base | 95 document(s), 4729 chunks (~110 words, 1-sentence overlap) |
| Retrieval | dense + keyword fusion (alpha=0.5), top-k=3 |
| Refusal threshold | 0.25 |
| Prompt version | v1 |
| Decoding | greedy (temperature 0), max_new_tokens=200 |

### 3. Token size
| Metric | Value |
|---|---|
| Prompt tokens per query (avg / max) | 0 (no LLM) |
| Output tokens per answer (avg / max) | 0 (no LLM) |
| Context window used (avg) | n/a |

### 4. Speed
| Metric | Value |
|---|---|
| Retrieval latency (avg) | 8 ms |
| Generation latency (avg, LLM calls only) | n/a |
| End-to-end latency (avg / p95) | 0.01 s / 0.01 s |
| Throughput | n/a |

### 5. Answer quality
| Metric | Value |
|---|---|
| Questions (answerable / not in documents) | 398 / 40 |
| Chunk recall@1 (gold chunk retrieved) | 240/398 (60%) |
| Chunk recall@3 (gold chunk retrieved) | 316/398 (79%) |
| Chunk recall@5 (gold chunk retrieved) | 334/398 (84%) |
| MRR@5 | 0.700 |
| Document hit@3 (right document retrieved) | 368/398 (92%) |
| Answer accuracy (key facts present) | 299/398 (75%) |
| Reference-answer word recall (answered Qs) | 0.60 |
| Correct refusals (not-in-documents Qs) | 8/40 (20%) |
| Hallucination rate (answered a not-in-documents Q) | 32/40 (80%) |
| Valid questions wrongly refused | 0/398 (0%) |
| Mean grounding score | 1.00 |
| Answers with citations | 430/430 (100%) |