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


## Combine cross-encoder with feature reranker -> hybrid neural reranker

resume
  ↓
Embedding Retriever
  ↓
Top-10 candidates
  ↓
Feature Scorer + Cross-Encoder Scorer
  ↓
Hybrid Neural Reranker
  ↓
Final ranking


# Final model comparison on the extended validation set

| Model | Hit@1 | Hit@3 | Hit@5 |
|------|------:|------:|------:|
| TF-IDF Baseline | 0.833 | 1.000 | 1.000 |
| Structured TF-IDF | 0.833 | 1.000 | 1.000 |
| Hybrid TF-IDF | 0.833 | 1.000 | 1.000 |
| Embedding Retriever | 0.722 | 0.944 | 0.944 |
| Two-stage + Feature Reranker | 0.833 | 0.944 | 1.000 |
| Two-stage + Cross-Encoder Reranker | 0.833 | 0.889 | 1.000 |
| **Two-stage + Hybrid Neural Reranker** | **0.833** | **1.000** | **1.000** |

## Current best architecture

The best-performing current setup is a **two-stage hybrid neural ranking pipeline**:

1. **Embedding-based retrieval** (`all-MiniLM-L6-v2`
2. **Hybrid neural reranking**, combining:
   - retrieval score
   - cross-encoder score
   - feature-based bonuses scores

This model preserves the semantic strength of neural reranking while retaining domain-specific signals useful for distinguishing closely related role families.

## Key finding

The experiments showed that neither a pure feature-based reranker nor a pure neural reranker was sufficient on its own.

The best results were achieved by **combining**:
- semantic retrieval,
- neural pairwise reranking,
- structured domain features.

## Remaining failure mode

The main remaining error pattern is confusion between:

- `bi`
- `analytics`

especially when resumes contain overlapping terms such as:
- dashboards
- KPI tracking
- reporting
- business analytics


## Learned reranker experiment

A trainable pointwise reranker was trained on top of the two-stage retrieval pipeline.

### Training setup

The learned reranker uses the following features for each `(resume, job)` pair:

- normalized embedding retrieval score
- normalized cross-encoder score
- skill overlap bonus
- domain phrase bonus
- title alignment bonus

A simple **Logistic Regression** model was trained as a pointwise reranker.

### Training data

The reranker was trained on synthetic query-job pairs generated from the training split:

- pair rows: **180**
- positive labels: **100**
- negative labels: **80**

### Validation results

| Metric | Value |
|------|------:|
| Hit@1 | 0.889 |
| Hit@3 | 1.000 |
| Hit@5 | 1.000 |

### Test results

| Metric | Value |
|------|------:|
| Hit@1 | 0.944 |
| Hit@3 | 0.944 |
| Hit@5 | 1.000 |

### Interpretation

The learned reranker achieved the best **top-1 ranking quality** among all tested models on the current test set.

It outperformed manually weighted rerankers in Hit@1, which suggests that learning how to combine retrieval, neural, and domain-specific signals is more effective than setting these weights by hand.

### Remaining failure mode

The main remaining error is still confusion between:

- `analytics`
- `bi`

This indicates that the main challenge of the task is ranking closely related analytical roles rather than retrieving obviously relevant jobs.


## Current-best pipeline:

Embedding retriever + learned reranker trained on hard-negative pairs

Without hard-negative pairing, distribution was 95%+ positive. Which made the model train on "almost all pairs are positive",
with this change, the model is trained on balanced pairs with hard-negatives. Hard-negatives were chosen instead of random-negatives,
because random-negatives are too easy.

### Learned reranker (linear regression) interpretation

Inspection of the logistic regression coefficients showed that the strongest signals for relevance were:

- skill overlap
- normalized retrieval score
- title alignment
- normalized cross-encoder score

This suggests that the learned reranker relies most on a combination of:
- structured skill matching,
- strong retrieval candidates,
- role-family alignment,
- and semantic pairwise relevance.

Interestingly, the domain phrase bonus received a slightly negative coefficient, which suggests that it may be partially redundant with other features or too noisy in the current dataset.

### Ablation study

An ablation experiment was run to test whether `domain_phrase_bonus` improves the learned reranker.

Result:
- removing `domain_phrase_bonus` did **not** change validation or test metrics
- the same single test error remained
- therefore, the final learned reranker excludes this feature

This suggests that `domain_phrase_bonus` was redundant relative to stronger signals such as:
- skill overlap
- retrieval score
- title alignment
- cross-encoder score
