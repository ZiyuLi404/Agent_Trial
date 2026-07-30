# History borrowing algorithms

The implementations are separated by method while continuing to share the
existing inputs and generated-data area under `history_borrowing/data/`.

## Layout

- `algorithm_1/` — convex similarity-aware history borrowing
- `algorithm_2/` — Bayesian pseudo-posterior history borrowing
- `data/` — shared ground-truth inputs, similarity matrices, and results

## Algorithm 1

See [`algorithm_1/README.md`](algorithm_1/README.md) for its workflow and CLI
commands.

## Algorithm 2

Compute the full 100-case reference posterior for each model/version:

```bash
python3 history_borrowing/algorithm_2/full_posterior.py
```

This writes
`history_borrowing/data/results/algorithm_2/full_posterior.csv` using
`Beta(1 + n_full * accuracy_full, 1 + n_full * (1 - accuracy_full))`.
Use `--alpha0` and `--beta0` to change the prior.

```bash
python3 history_borrowing/algorithm_2/bayesian_pseudo_posterior.py \
  --data_dir history_borrowing/data/groundtruth \
  --similarity_file history_borrowing/data/similarity_matrix/embedding_diagnosis_similarity_matrix.csv \
  --lambda 10 \
  --alpha0 1 \
  --beta0 1 \
  --credible_level 0.95 \
  --all_orders \
  --output_dir history_borrowing/data/results/bayesian_pseudo_posterior
```

The trajectory records the predictive and updated Beta parameters, means,
variances, and equal-tailed pseudo-posterior intervals. These intervals describe
the pseudo-posterior but should not be interpreted as calibrated until the
borrowing rule has been validated.

With `--all_orders`, Algorithm 2 also writes `bayesian_update_all_orders.csv`,
the model/order summaries, `best_orders_by_wasserstein.csv`, and
`best_orders_by_mae.csv`. The reference is the paper's full-data Beta posterior;
it is not treated as exact ground truth. Omit the flag when only the sequential
Bayesian trajectory is needed.

Run its unit tests with:

```bash
python3 -m unittest history_borrowing/algorithm_2/test_bayesian_pseudo_posterior.py -v
```
