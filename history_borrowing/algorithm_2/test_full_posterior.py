"""Unit tests for the full-data reference posterior computation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from history_borrowing.algorithm_2.bayesian_pseudo_posterior import (
    ModelOutcomes,
    load_full_model_outcomes,
)
from history_borrowing.algorithm_2.full_posterior import (
    FIELDNAMES,
    build_full_posterior_rows,
    compute_full_posterior,
    write_full_posterior,
)


class FullPosteriorTests(unittest.TestCase):
    def test_computes_beta_summary_from_all_cases(self):
        outcome = ModelOutcomes("model-a", (1.0, 1.0, 0.0, 1.0))

        row = compute_full_posterior(outcome)

        self.assertEqual(row["n_full"], 4)
        self.assertEqual(row["accuracy_full"], 0.75)
        self.assertEqual(row["alpha_full"], 4.0)
        self.assertEqual(row["beta_full"], 2.0)
        self.assertAlmostEqual(row["posterior_mean"], 4.0 / 6.0)
        self.assertAlmostEqual(row["posterior_variance"], 8.0 / (36.0 * 7.0))
        self.assertLess(row["lower_95_ci"], row["posterior_mean"])
        self.assertGreater(row["upper_95_ci"], row["posterior_mean"])

    def test_loads_existing_flat_and_nested_correctness_formats(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            data_dir = Path(temporary_dir)
            records = [
                {"correct": True},
                {"evaluation": {"is_correct": "incorrect"}},
                {"result": {"score": 1}},
            ]
            (data_dir / "model-version.json").write_text(
                json.dumps({"results": records}), encoding="utf-8"
            )

            outcomes = load_full_model_outcomes(data_dir)

        self.assertEqual(outcomes, [ModelOutcomes("model-version", (1.0, 0.0, 1.0))])

    def test_requires_100_cases_by_default(self):
        with self.assertRaisesRegex(ValueError, "expected 100"):
            build_full_posterior_rows([ModelOutcomes("incomplete", (1.0,) * 99)])

    def test_writes_only_required_columns(self):
        rows = build_full_posterior_rows(
            [ModelOutcomes("model-a", (1.0, 0.0))], expected_n=2
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "full_posterior.csv"
            write_full_posterior(output_path, rows)
            with output_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                written_rows = list(reader)

        self.assertEqual(reader.fieldnames, FIELDNAMES)
        self.assertEqual(len(written_rows), 1)
        self.assertNotIn("mae", written_rows[0])

    def test_rejects_nonpositive_prior_parameters(self):
        outcome = ModelOutcomes("model-a", (1.0,))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            compute_full_posterior(outcome, alpha0=0.0)


if __name__ == "__main__":
    unittest.main()
