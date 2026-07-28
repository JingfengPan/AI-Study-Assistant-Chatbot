# Evaluation Results

- Date: 2026-07-28T07:40:25.022667+00:00
- Chat/judge model: `gpt-5.6-terra`
- Embedding model: `text-embedding-3-small`
- Answer prompt: `rag-answer-v1`
- Stuffing prompt: `stuffing-baseline-v1`
- Judge prompt: `rag-judge-v1`
- Chunk size / overlap: 600 / 100 tokens
- Top-k: 4
- Dataset: 30 questions

| Mode | Recall@1 | Recall@3 | Recall@4 | Answer Accuracy | Citation Source Accuracy | Refusal Accuracy | False Refusal Rate | Avg. Input Tokens | Median Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stuffing | N/A | N/A | N/A | 53.3% | N/A | 100.0% | 12.0% | 700.2 | 1,048.0 ms |
| Rag | 100.0% | 100.0% | 100.0% | 96.7% | 100.0% | 100.0% | 0.0% | 455.3 | 1,425.2 ms |

## Supplemental judge metrics

- **Stuffing**: correctness N/A, groundedness N/A, relevance N/A; 0 judge failures.
- **Rag**: correctness 100.0%, groundedness 100.0%, relevance 100.0%; 0 judge failures.

## Failure examples

### Stuffing · q001

**Question:** What prerequisite does binary search have, and how does it reduce the search space?

**Answer:** Binary search requires sorted data. It compares against the middle element and repeatedly halves the remaining search range.

Reference match: False; Recall@4: N/A

### Stuffing · q005

**Question:** How does separate chaining handle hash collisions?

**Answer:** Separate chaining stores multiple colliding entries in the same hash-table bucket, typically using a linked list or similar collection.

Reference match: False; Recall@4: N/A

### Stuffing · q007

**Question:** What dependency does third normal form remove?

**Answer:** Third normal form removes transitive dependencies.

Reference match: False; Recall@4: N/A

### Stuffing · q008

**Question:** What four properties make up ACID?

**Answer:** I could not find sufficient supporting information in the uploaded materials.

Reference match: False; Recall@4: N/A

### Stuffing · q009

**Question:** What operations does a B-tree index support efficiently?

**Answer:** B-tree indexes support efficient searches and range queries.

Reference match: False; Recall@4: N/A

### Rag · q005

**Question:** How does separate chaining handle hash collisions?

**Answer:** Separate chaining handles collisions by storing all keys that hash to the same bucket in a per-bucket collection. [1]

Sources:
[1]

Reference match: False; Recall@4: True

## Limitations

- The corpus and golden set are intentionally small and synthetic.
- Reference-match accuracy is a deterministic lexical metric; supplemental judge scores are model-based and not ground truth.
- Citation syntax and source matching do not prove that every claim is entailed by its cited passage.
- Results depend on the selected models, prompts, chunking, and API behavior at the recorded date.
