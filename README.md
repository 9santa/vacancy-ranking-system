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
