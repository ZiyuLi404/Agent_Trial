import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_system_adapters.agentclinic import AgentClinicAdapter
from change_generators.skills.optimizers.skillopt_lite.evaluator import (
    build_parser,
    evaluate,
)
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
                "trajectory",
                "variant",
            }
            for row in rows:
                self.assertTrue(required.issubset(row))
                self.assertEqual(row["phase"], "contract_dry_run")
                prediction_dir = output_dir / "predictions" / row["id"]
                self.assertTrue((prediction_dir / "conversation.json").is_file())
                self.assertTrue((prediction_dir / "target_system_prompt.txt").is_file())
                self.assertTrue((prediction_dir / "target_user_prompt.txt").is_file())

            failed_samples = list(
                (workspace / ".skillopt" / "samples" / "failed").glob("*.md")
            )
            self.assertEqual(len(failed_samples), 2)


if __name__ == "__main__":
    unittest.main()
