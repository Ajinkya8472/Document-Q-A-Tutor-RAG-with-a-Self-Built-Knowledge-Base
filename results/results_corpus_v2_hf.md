# Benchmark: hf

### 1. LLM used (model & parameters)
| Metric | Value |
|---|---|
| Backend / model | hf / Qwen/Qwen2.5-1.5B-Instruct |
| Model parameters | 1543.7 M |
| Model size in memory | 3087 MB (float16) on cuda |
| Context window (max tokens) | 32768 |
| Embedding model | all-MiniLM-L6-v2, 22.7 M params, 384-dim |

### 2. Pipeline hyper-parameters
| Metric | Value |
|---|---|
| Knowledge base | 97 document(s), 4913 chunks (~110 words, 1-sentence overlap) |
| Retrieval | dense + keyword fusion (alpha=0.5), top-k=3 |
| Refusal threshold | 0.25 |
| Prompt version | v2 |
| Decoding | greedy (temperature 0), max_new_tokens=200 |

### 3. Token size
| Metric | Value |
|---|---|
| Prompt tokens per query (avg / max) | 528 / 881 |
| Output tokens per answer (avg / max) | 33 / 90 |
| Context window used (avg) | 1.6% |

### 4. Speed
| Metric | Value |
|---|---|
| Retrieval latency (avg) | 37 ms |
| Generation latency (avg, LLM calls only) | 3.52 s |
| End-to-end latency (avg / p95) | 3.51 s / 7.71 s |
| Throughput | 9.2 tokens/s |

### 5. Answer quality
| Metric | Value |
|---|---|
| Questions (answerable / not in documents) | 398 / 40 |
| Chunk recall@1 (gold chunk retrieved) | 240/398 (60%) |
| Chunk recall@3 (gold chunk retrieved) | 316/398 (79%) |
| Chunk recall@5 (gold chunk retrieved) | 334/398 (84%) |
| MRR@5 | 0.700 |
| Document hit@3 (right document retrieved) | 368/398 (92%) |
| Answer accuracy (key facts present) | 331/398 (83%) |
| Reference-answer word recall (answered Qs) | 0.69 |
| Correct refusals (not-in-documents Qs) | 19/40 (48%) |
| Hallucination rate (answered a not-in-documents Q) | 21/40 (52%) |
| Valid questions wrongly refused | 4/398 (1%) |
| Mean grounding score | 0.73 |
| Answers with citations | 2/415 (0%) |