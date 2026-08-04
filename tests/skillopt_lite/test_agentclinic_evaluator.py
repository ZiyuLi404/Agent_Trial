import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_system_adapters.agentclinic import AgentClinicAdapter
from change_generators.skills.optimizers.skillopt_lite.evaluator import (
    build_result_row,
    build_parser,
    evaluate,
)
from change_generators.skills.optimizers.skillopt_lite.gate import compare_result_sets
from change_generators.skills.optimizers.skillopt_lite.samples import export_samples
from change_generators.skills.optimizers.skillopt_lite.splits import (
    build_split_manifest,
    load_split_config,
    select_case_ids,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT
    / "change_generators"
    / "skills"
    / "artifacts"
    / "diagnostic_reasoning"
    / "v000.md"
)
SPLIT_CONFIG = REPO_ROOT / "experiments" / "agentclinic_skillopt" / "splits.toml"
CLEAN_SPLIT_CONFIG = (
    REPO_ROOT / "experiments" / "agentclinic_skillopt" / "splits_clean_v1.toml"
)


class SkillOptLiteSplitTest(unittest.TestCase):
    def test_medqa_split_is_deterministic_disjoint_and_complete(self):
        config = load_split_config(SPLIT_CONFIG)
        first = build_split_manifest("MedQA", 107, config)
        second = build_split_manifest("MedQA", 107, config)

        self.assertEqual(first["splits"], second["splits"])
        self.assertEqual(len(first["splits"]["train"]), 21)
        self.assertEqual(len(first["splits"]["val"]), 21)
        self.assertEqual(len(first["splits"]["test"]), 65)

        split_sets = [set(first["splits"][name]) for name in ("train", "val", "test")]
        self.assertTrue(split_sets[0].isdisjoint(split_sets[1]))
        self.assertTrue(split_sets[0].isdisjoint(split_sets[2]))
        self.assertTrue(split_sets[1].isdisjoint(split_sets[2]))
        self.assertEqual(set.union(*split_sets), set(range(107)))

        sample = select_case_ids(first, "train", eval_limit=3, sample_seed=17)
        self.assertEqual(
            sample,
            select_case_ids(first, "train", eval_limit=3, sample_seed=17),
        )

    def test_expected_dataset_size_is_guarded(self):
        config = load_split_config(SPLIT_CONFIG)
        with self.assertRaisesRegex(ValueError, "split config expects 107"):
            build_split_manifest("MedQA", 106, config)

    def test_clean_protocol_excludes_only_exposed_validation_cases(self):
        standard = build_split_manifest(
            "MedQA", 107, load_split_config(SPLIT_CONFIG)
        )
        clean = build_split_manifest(
            "MedQA", 107, load_split_config(CLEAN_SPLIT_CONFIG)
        )
        exposed = {56, 68, 80, 30, 44}

        self.assertEqual(set(standard["splits"]["val"]) - exposed,
                         set(clean["splits"]["val"]))
        self.assertEqual(len(clean["splits"]["val"]), 16)
        self.assertEqual(set(clean["excluded_ids"]["val"]), exposed)
        self.assertEqual(clean["protocol"]["id"], "clean_v1")


class SkillOptLiteSampleTest(unittest.TestCase):
    def test_samples_are_sorted_into_passed_and_failed_directories(self):
        rows = [
            {"id": "pass-1", "hard": 1, "soft": 1.0, "split": "train"},
            {
                "id": "fail-1",
                "hard": 0,
                "soft": 0.0,
                "split": "train",
                "fail_reason": "incorrect",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            counts = export_samples(rows, temp_dir)
            root = Path(temp_dir) / ".skillopt" / "samples"

            self.assertEqual(counts, {"failed": 1, "passed": 1})
            self.assertTrue((root / "passed" / "pass-1.md").is_file())
            self.assertTrue((root / "failed" / "fail-1.md").is_file())


class SkillOptLiteGateTest(unittest.TestCase):
    def test_flat_paired_gate_retains_baseline_and_reports_case_swaps(self):
        baseline = {
            "a": {"id": "a", "hard": 0, "correct_text": "A", "predicted_answer": "x"},
            "b": {"id": "b", "hard": 1, "correct_text": "B", "predicted_answer": "B"},
        }
        candidate = {
            "a": {"id": "a", "hard": 1, "correct_text": "A", "predicted_answer": "A"},
            "b": {"id": "b", "hard": 0, "correct_text": "B", "predicted_answer": "y"},
        }

        report = compare_result_sets(baseline, candidate)

        self.assertEqual(report["action"], "flat")
        self.assertEqual(report["selected"], "baseline")
        self.assertEqual(report["improved_cases"], 1)
        self.assertEqual(report["regressed_cases"], 1)
        self.assertEqual(report["delta"], 0.0)

    def test_paired_gate_rejects_mismatched_ids(self):
        baseline = {"a": {"id": "a", "hard": 1}}
        candidate = {"b": {"id": "b", "hard": 1}}
        with self.assertRaisesRegex(ValueError, "identical ids"):
            compare_result_sets(baseline, candidate)

    def test_paired_gate_excludes_infrastructure_invalid_pairs(self):
        baseline = {
            "valid": {"id": "valid", "hard": 1, "correct_text": "A"},
            "noisy": {
                "id": "noisy",
                "hard": 0,
                "correct_text": "B",
                "agent_ok": False,
            },
        }
        candidate = {
            "valid": {"id": "valid", "hard": 1, "correct_text": "A"},
            "noisy": {"id": "noisy", "hard": 1, "correct_text": "B"},
        }

        report = compare_result_sets(baseline, candidate)

        self.assertEqual(report["n"], 1)
        self.assertEqual(report["n_total"], 2)
        self.assertEqual(report["n_excluded"], 1)
        self.assertEqual(report["delta"], 0.0)


class EvaluatorTelemetryTest(unittest.TestCase):
    class Scenario:
        def examiner_information(self):
            return "Diagnose this case."

        def diagnosis_information(self):
            return "Diagnosis A"

    def test_result_row_preserves_retry_health_and_observed_calls(self):
        row, _ = build_result_row(
            dataset="MedQA",
            split="val",
            case_id=1,
            scenario=self.Scenario(),
            diagnosis="DIAGNOSIS READY: Diagnosis A",
            correctness=True,
            dialogue=(
                "Doctor: Question one\n"
                "Patient: Answer one\n"
                "Doctor: DIAGNOSIS READY: Diagnosis A\n"
            ),
            meta={
                "raw_doctor_response_empty": True,
                "doctor_retry_count": 1,
                "reasoning_content_present": True,
            },
            contract_dry_run=False,
        )

        self.assertEqual(row["observed_model_calls"], 5)
        self.assertEqual(row["backend"]["doctor_retry_count"], 1)
        self.assertTrue(row["backend"]["reasoning_content_present"])

    def test_grounded_measurement_is_not_counted_as_a_model_call(self):
        row, _ = build_result_row(
            dataset="MedQA",
            split="val",
            case_id=1,
            scenario=self.Scenario(),
            diagnosis="DIAGNOSIS READY: Diagnosis A",
            correctness=True,
            dialogue=(
                "Doctor: REQUEST TEST: CBC\n"
                "Measurement: RESULTS: source-backed CBC\n"
                "Doctor: DIAGNOSIS READY: Diagnosis A\n"
            ),
            meta={"measurement": {"mode": "grounded_lookup"}},
            contract_dry_run=False,
        )

        self.assertEqual(row["observed_model_calls"], 3)


class AgentClinicSkillOptEvaluatorContractTest(unittest.TestCase):
    def test_contract_dry_run_writes_complete_contract_without_api_calls(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            workspace = temp_root / "workspace"
            output_dir = temp_root / "output"
            args = build_parser().parse_args(
                [
                    "--skill",
                    str(SKILL_PATH),
                    "--split_config",
                    str(SPLIT_CONFIG),
                    "--dataset",
                    "MedQA",
                    "--split",
                    "train",
                    "--eval_limit",
                    "2",
                    "--limit",
                    "2",
                    "--workspace",
                    str(workspace),
                    "--output_dir",
                    str(output_dir),
                    "--contract_dry_run",
                ]
            )

            with patch.object(
                AgentClinicAdapter,
                "evaluate_case",
                side_effect=AssertionError("dry-run attempted an API-backed evaluation"),
            ) as evaluate_case:
                summary = evaluate(args)

            evaluate_case.assert_not_called()
            self.assertEqual(summary["n"], 2)
            self.assertEqual(summary["hard"], 0.0)
            self.assertEqual(summary["soft"], 0.0)
            self.assertTrue((workspace / "skill.md").is_file())
            self.assertTrue((output_dir / "metrics.json").is_file())
            self.assertTrue((output_dir / "split_manifest.json").is_file())

            rows = [
                json.loads(line)
                for line in (output_dir / "results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 2)
            required = {
                "id",
                "hard",
                "soft",
                "question",
                "correct_text",
                "predicted_answer",
                "fail_reason",
                "n_turns",
                "tests_requested",
                "observed_model_calls",
                "backend",
                "trajectory",
                "variant",
            }
            for row in rows:
                self.assertTrue(required.issubset(row))
                self.assertEqual(row["phase"], "contract_dry_run")
                self.assertEqual(row["observed_model_calls"], 0)
                prediction_dir = output_dir / "predictions" / row["id"]
                self.assertTrue((prediction_dir / "conversation.json").is_file())
                self.assertTrue((prediction_dir / "target_system_prompt.txt").is_file())
                self.assertTrue((prediction_dir / "target_user_prompt.txt").is_file())

            failed_samples = list(
                (workspace / ".skillopt" / "samples" / "failed").glob("*.md")
            )
            self.assertEqual(len(failed_samples), 2)
            self.assertEqual(summary["observed_model_calls"], 0)
            self.assertEqual(summary["backend_health"]["total_doctor_retries"], 0)


if __name__ == "__main__":
    unittest.main()
