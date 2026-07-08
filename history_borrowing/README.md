# history_borrowing — History Borrowing / Performance Estimation (H)

**When a model has too few cases in a bucket to trust its accuracy, borrow data from "models that behave similarly" to correct the estimate — so a small sample can still be estimated well.**

Offline analysis — no consultations, no engine. Reads accuracy and similarity data that were already produced.

## Inputs
- `accuracy_by_25_cases.csv` — per-bucket accuracy + gold counts (from `accuracy_summary.py`)
- `*_similarity_matrix.csv` — model-to-model similarity (from **`embedding_similarity` (G)**)

## How it works
Turn similarity into distance `d(A,B) = 1 - sim(A,B)`, then let each model borrow its peers' accuracy, distance-weighted:

```
theta_borrowed_j = alpha * theta_j + (1 - alpha) * Σ_{i≠j} w_ij * theta_i
```

`alpha` trades off "trust myself" vs "trust peers"; `w_ij` comes from distance via `lambda`.

## Files (5 = 5 real steps, kept separate)
| File | Step |
|------|------|
| `accuracy_summary.py` | Summarize per-bucket accuracy → `accuracy_by_25_cases.csv` |
| `history_borrowing.py` | One borrowing estimate (given alpha/lambda) |
| `run_all_orders.py` | Run `history_borrowing.py` over all 24 bucket↔model permutations |
| `train_borrow_params.py` | Fit one global (alpha, lambda) minimizing mean MAE |
| `visualize_borrow_params.py` | Render the results as a dashboard |

## Run
```bash
# from the repo root
python history_borrowing/accuracy_summary.py --groundtruth_dir history_borrowing/data/groundtruth
python history_borrowing/history_borrowing.py --accuracy_csv ... --similarity_csv ...
python history_borrowing/run_all_orders.py
python history_borrowing/train_borrow_params.py
python history_borrowing/visualize_borrow_params.py --source diagnosis
```

Current data lives under `history_borrowing/data/`, and the scripts default to that layout.
Generated result files are written under `history_borrowing/data/results/`.
Use `--similarity_csv history_borrowing/data/similarity_matrix/fingerprint_conversation_similarity.csv`
to evaluate the fingerprint conversation similarity matrix.

## Relationship
- Upstream: similarity matrices from **`embedding_similarity` (G)**.
- Sibling of **`deployment_replay` (E)**: both are "a new version has too little data — borrow from the past." H borrows *horizontally* (similar models), E borrows *vertically* (its own old cases).

## Notes
- ✅ Renamed `performance_estimation → history_borrowing`; internal self-paths updated.
- Current inputs are kept under `history_borrowing/data/`; generated outputs go under `history_borrowing/data/results/`.
