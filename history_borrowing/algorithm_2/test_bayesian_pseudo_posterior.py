"""Small unit tests for Algorithm 2's Bayesian equations."""

import math
import unittest
from pathlib import Path

from history_borrowing.algorithm_2.bayesian_pseudo_posterior import (
    ModelOutcomes,
    beta_credible_interval,
    beta_wasserstein_1,
    borrowing_weight,
    build_all_orders_report,
    construct_history_prior,
    online_posterior_trajectory,
)


class BayesianPseudoPosteriorTests(unittest.TestCase):
    def test_history_contributes_fractional_prior_evidence(self):
        history = [ModelOutcomes("old", (1.0, 0.0, 1.0, 0.0))]
        similarities = {
            "old": {"new": 0.8},
            "new": {"new": 1.0},
        }
        alpha, beta, rows = construct_history_prior(
            "new", history, similarities, math.log(2.0) / 0.2, 1.0, 1.0
        )

        self.assertAlmostEqual(rows[0]["weight"], 0.5)
        self.assertAlmostEqual(alpha, 2.0)  # 1 + 0.5 * 4 * 0.5
        self.assertAlmostEqual(beta, 2.0)   # 1 + 0.5 * 4 * (1 - 0.5)
        self.assertLess(rows[0]["borrowed_alpha_evidence"], 2.0)

    def test_higher_similarity_has_larger_weight(self):
        high = borrowing_weight(0.9, 1.0, 10.0)
        low = borrowing_weight(0.5, 1.0, 10.0)
        self.assertGreater(high, low)

    def test_posterior_mean_updates_after_outcomes(self):
        rows = online_posterior_trajectory("new", [1.0, 0.0], 2.0, 2.0)
        self.assertAlmostEqual(rows[0]["predicted_accuracy"], 0.5)
        self.assertAlmostEqual(rows[0]["posterior_mean"], 3.0 / 5.0)
        self.assertAlmostEqual(rows[1]["posterior_mean"], 3.0 / 6.0)
        self.assertEqual((rows[1]["alpha"], rows[1]["beta"]), (3.0, 3.0))
        self.assertLess(
            rows[0]["posterior_interval_lower"], rows[0]["posterior_mean"]
        )
        self.assertGreater(
            rows[0]["posterior_interval_upper"], rows[0]["posterior_mean"]
        )

    def test_beta_distribution_interval_and_wasserstein(self):
        lower, upper = beta_credible_interval(1.0, 1.0, 0.95)
        self.assertAlmostEqual(lower, 0.025, places=6)
        self.assertAlmostEqual(upper, 0.975, places=6)
        self.assertAlmostEqual(beta_wasserstein_1(2.0, 3.0, 2.0, 3.0), 0.0)

    def test_all_orders_report_contains_every_order_and_step(self):
        buckets = {
            "old": [ModelOutcomes("old", (1.0, 0.0)), ModelOutcomes("old", (1.0, 1.0))],
            "new": [ModelOutcomes("new", (0.0, 0.0)), ModelOutcomes("new", (1.0, 0.0))],
        }
        similarities = {
            "old": {"old": 1.0, "new": 0.8},
            "new": {"old": 0.8, "new": 1.0},
        }
        detail, by_model, by_order, best = build_all_orders_report(
            buckets, similarities, Path("embedding.csv"), 1.0, 1.0, 1.0
        )

        self.assertEqual(len(detail), 4)  # 2! orders * 2 deployment steps
        self.assertEqual(len(by_model), 2)
        self.assertEqual(len(by_order), 2)
        self.assertEqual([row["rank"] for row in best], [1, 2])
        self.assertIn("mean_wasserstein_1", by_order[0])


if __name__ == "__main__":
    unittest.main()
