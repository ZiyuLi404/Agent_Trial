"""
Hybrid estimator for deployment_replay mode.

Computes the practical hybrid performance estimate for a new treatment model:
a weighted average of historical-replay accuracy (on past cases) and concurrent
accuracy (on new cases in the current epoch), weighted by case counts.

This is what can be observed *without* calling back past patients.  Comparing it
to the oracle (which re-runs the new model interactively on all past cases) tells
us how much information is lost when we substitute transcript replay for live
re-interaction.

Design note
-----------
Both hybrid and oracle use the same concurrent-treatment results for new cases.
The only difference is how past cases are evaluated:
  hybrid  — past cases evaluated via transcript replay
  oracle  — past cases evaluated via fresh interactive re-runs
This keeps their denominators identical, making the comparison direct.
"""


def compute_hybrid_estimate(replay_results, concurrent_results):
    """Weighted accuracy: (replay correct + concurrent correct) / total cases.

    Parameters
    ----------
    replay_results : list[dict]
        Records from historical_replay evaluation; each must have 'correctness'.
    concurrent_results : list[dict]
        Records from concurrent evaluation (treatment arm only) in the current
        epoch; each must have 'correctness'.

    Returns
    -------
    float
        Weighted accuracy in [0.0, 1.0].  Returns 0.0 if both lists are empty.
    """
    total = len(replay_results) + len(concurrent_results)
    if total == 0:
        return 0.0

    correct = sum(1 for r in replay_results if r["correctness"]) + \
              sum(1 for r in concurrent_results if r["correctness"])
    return correct / total


def compute_oracle_estimate(oracle_results, concurrent_results):
    """Weighted accuracy using oracle re-interaction on past cases.

    Mirrors compute_hybrid_estimate but substitutes oracle results for replay
    results, giving the ground-truth combined accuracy.

    Parameters
    ----------
    oracle_results : list[dict]
        Records from oracle_full_replay evaluation on past cases.
    concurrent_results : list[dict]
        Same concurrent_results passed to compute_hybrid_estimate.

    Returns
    -------
    float
        Oracle accuracy in [0.0, 1.0].
    """
    total = len(oracle_results) + len(concurrent_results)
    if total == 0:
        return 0.0

    correct = sum(1 for r in oracle_results if r["correctness"]) + \
              sum(1 for r in concurrent_results if r["correctness"])
    return correct / total
