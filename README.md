## Experiments and Baselines

Compared several retrieval baselines for the resume-to-job matching task.

### Evaluation setup

Current offline evaluation is performed on a small synthetic validation set of 6 resume queries with manually assigned target role families:

- `data_science`
- `analytics`
- `bi`

Metrics:

- **Hit@1** — whether the correct role family appears at rank 1
- **Hit@3** — whether the correct role family appears in top-3
- **Hit@5** — whether the correct role family appears in top-5

### Compared models

| Version | Model | Description | Hit@1 | Hit@3 | Hit@5 |
|--------:|-------|-------------|------:|------:|------:|
| v1 | `tfidf_v1` | Baseline TF-IDF over a single combined job text | 0.833 | 1.000 | 1.000 |
| v2 | `structured_tfidf_v2` | Separate TF-IDF vectorization & scoring for title, skills, and description | 0.833 | 1.000 | 1.000 |
| v3 | `hybrid_tfidf_v3` | Structured TF-IDF + rule-based bonus features | 0.833 | 1.000 | 1.000 |
| v4 | `embeddings_v4` | Semantic retrieval with `sentence-transformers/all-MiniLM-L6-v2` | 1.000 | 1.000 | 1.000 |



### Results on the extended evaluation set

| Model | Hit@1 | Hit@3 | Hit@5 |
|------|------:|------:|------:|
| Embedding Retriever (`all-MiniLM-L6-v2`) | 0.722 | 0.944 | 0.944 |

### Metrics by difficulty

| Difficulty | Hit@1 | Hit@3 | Hit@5 |
|-----------|------:|------:|------:|
| easy | 0.667 | 1.000 | 1.000 |
| medium | 0.667 | 1.000 | 1.000 |
| hard | 0.833 | 0.833 | 0.833 |

Embeddings significantly improved retrieval quality compared to lexical baselines, but harder evaluation dataset showed that semantic retrieval alone is still not sufficient for top-1 ranking on closely related analytical roles.

This suggests a two-stage approach: embedding-based retrieval + reranking.

resume_text
   ↓
Embedding Retriever
   ↓
Top-10 candidate jobs
   ↓
Reranker
   ↓
Final top-5 jobs


### Results

| Model | Hit@1 | Hit@3 | Hit@5 |
|------|------:|------:|------:|
| Embedding Retriever (`all-MiniLM-L6-v2`) | 0.722 | 0.944 | 0.944 |
| Two-Stage Matcher (Embeddings + Feature Reranker) | 0.833 | 0.944 | 1.000 |

The two-stage setup improved final ranking quality over the pure embedding retriever:

- **Hit@1 improved from 0.722 to 0.833**
- **Hit@5 improved from 0.944 to 1.000**
- **Hit@3 remained unchanged at 0.944**

## Cross-encoder reranker comes to play

resume
  ↓
embedding retriever
  ↓
top-10 jobs
  ↓
cross-encoder reranker
  ↓
final ranking
