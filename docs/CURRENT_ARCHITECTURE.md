# Current Best Architecture

## Current best model

The current best pipeline is:

1. Embedding-based retrieval using `all-MiniLM-L6-v2`
2. Learned reranker trained over:
   - normalized retrieval score
   - normalized cross-encoder score
   - skill overlap bonus
   - domain phrase bonus
   - title alignment bonus

## Main entry points

Inference:

```bash
python -m src.inference.recommend_current
```

Evaluation:

```bash
python -m src.evaluation.evaluate_current
```

Training the learned reranker:

```bash
python -m src.training.train_learned_reranker
```

## Legacy experiments

Older lexical, hybrid, and intermediate reranking experiments are preserved under:

- `src/models/legacy/`
- `src/inference/legacy/`
- `src/evaluation/legacy/`
