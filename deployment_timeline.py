"""
Simulated deployment timeline for --eval_mode deployment_replay.

Each epoch runs *paired shadow evaluation*: every new case is run by both the
fixed control model and the current treatment model.  This is not a randomized
RCT (randomization.py is untouched and used only by accuracy mode).  In a
simulation/shadow setting we want both models on the identical case stream so
comparisons are exact rather than statistically adjusted.

Per-epoch phases
----------------
Phase 1 — Paired concurrent: for each of N new cases, run control AND treatment.
           Log both records with the same case_id / paired_case_id.
           Persist the treatment transcript to disk for future replay.
Phase 2 — Replay + Oracle on all past treatment transcripts (epoch > 0 only):
           2a. Historical replay — new treatment diagnoses from saved transcript.
           2b. Oracle full replay — new treatment re-interacts from scratch.
Phase 3 — Metrics summary:
           total_n        = past_n + current_n
           control_acc    = control_correct_so_far / total_n
           hybrid_acc     = (replay_correct + current_treatment_correct) / total_n
           oracle_acc     = (oracle_past_correct + current_treatment_correct) / total_n
"""

import os
import json
import random
from datetime import datetime, timezone

from version_manager import open_new_version
from trial_manager import stream_cases, run_case
from replay_evaluator import run_replay_case
from oracle_evaluator import run_oracle_case
from logger import log_deployment_case


class DeploymentTimeline:
    """Orchestrates a multi-epoch deployment trial with paired shadow evaluation."""

    def __init__(
        self,
        control_model,
        treatment_schedule,
        epoch_sizes,
        dataset,
        total_inferences,
        output_dir,
        seed,
        patient_llm,
        measurement_llm,
        moderator_llm,
    ):
        """
        Parameters
        ----------
        control_model : str
            Fixed comparator model, run on every case in every epoch.
        treatment_schedule : list[str]
            Ordered treatment model names, one per epoch.
        epoch_sizes : list[int]
            Number of new cases per epoch (parallel to treatment_schedule).
        dataset : str
            AgentClinic dataset name, e.g. "MedQA".
        total_inferences : int
            Max doctor-patient turns per case.
        output_dir : str
            Root directory for all output files.
        seed : int
            Random seed (passed to random.seed for any internal shuffling).
        patient_llm, measurement_llm, moderator_llm : str
            Shared agent backends for every case.
        """
        self.control_model = control_model
        self.treatment_schedule = treatment_schedule
        self.epoch_sizes = epoch_sizes
        self.dataset = dataset
        self.total_inferences = total_inferences
        self.output_dir = output_dir
        self.seed = seed
        self.patient_llm = patient_llm
        self.measurement_llm = measurement_llm
        self.moderator_llm = moderator_llm

        self.shared_config = {
            "patient_llm": patient_llm,
            "measurement_llm": measurement_llm,
            "moderator_llm": moderator_llm,
            "total_inferences": total_inferences,
        }

        # Treatment concurrent records from past epochs — used for replay/oracle.
        self.past_records = []
        # Control concurrent records from past epochs — used for control_acc.
        self.past_control_records = []
        # Global case counter passed as start_id to stream_cases each epoch.
        self._global_start = 0

        os.makedirs(output_dir, exist_ok=True)
        random.seed(seed)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        """Run all epochs defined by treatment_schedule / epoch_sizes."""
        if len(self.treatment_schedule) != len(self.epoch_sizes):
            raise ValueError(
                "--treatment_schedule and --epoch_sizes must have the same length"
            )
        for epoch_idx, (treatment_model, n_cases) in enumerate(
            zip(self.treatment_schedule, self.epoch_sizes)
        ):
            print(f"\n{'=' * 60}")
            print(f"EPOCH {epoch_idx}  |  treatment={treatment_model}  |  new_cases={n_cases}")
            print(f"{'=' * 60}")
            self._run_epoch(epoch_idx, treatment_model, n_cases)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _run_epoch(self, epoch_idx, treatment_model, n_cases):
        epoch_id = f"epoch_{epoch_idx}"
        version = open_new_version(
            version_id=epoch_id,
            model_name=treatment_model,
            prompt_version="p1",
            tool_version="t1",
        )
        print(f"Opened version: {version}")

        transcript_dir = os.path.join(self.output_dir, "transcripts", epoch_id)
        os.makedirs(transcript_dir, exist_ok=True)

        control_config = {**self.shared_config, "doctor_llm": self.control_model}
        treatment_config = {**self.shared_config, "doctor_llm": treatment_model}

        new_epoch_treatment_records = []
        new_epoch_control_records = []

        # ── Phase 1: Paired shadow evaluation on N new cases ──────────────────
        # Both models run on every case. The arm field from stream_cases is
        # ignored — randomization.py is unchanged but not consulted here.
        print(f"\n[Phase 1] Paired shadow evaluation — {n_cases} new cases (both models)")

        for case_id, timestamp, scenario, _arm in stream_cases(
            self.dataset, n_cases, start_id=self._global_start
        ):
            ground_truth = str(scenario.diagnosis_information())

            # ── Control run ──────────────────────────────────────────────────
            ctrl_dx, ctrl_correct, ctrl_consult, ctrl_meta = run_case(scenario, control_config)
            ctrl_record = {
                "case_id": case_id,
                "paired_case_id": case_id,
                "dataset": self.dataset,
                "epoch_id": epoch_id,
                "version_id": version["version_id"],
                "doctor_model_name": self.control_model,
                "patient_llm": self.patient_llm,
                "arm": "control",
                "evaluation_type": "concurrent",
                "source_epoch": epoch_id,
                "timestamp": timestamp,
                "diagnosis": str(ctrl_dx),
                "correct_diagnosis": ground_truth,
                "correctness": ctrl_correct,
                "transcript_text": ctrl_consult,
                "run_id": 0,
                "random_seed": self.seed,
                **ctrl_meta,
            }
            log_deployment_case(ctrl_record, self.output_dir)
            new_epoch_control_records.append(ctrl_record)

            # ── Treatment run ────────────────────────────────────────────────
            tmt_dx, tmt_correct, tmt_consult, tmt_meta = run_case(scenario, treatment_config)
            tmt_record = {
                "case_id": case_id,
                "paired_case_id": case_id,
                "dataset": self.dataset,
                "epoch_id": epoch_id,
                "version_id": version["version_id"],
                "model_name": treatment_model,
                "arm": "treatment",
                "evaluation_type": "concurrent",
                "source_epoch": epoch_id,
                "timestamp": timestamp,
                "diagnosis": str(tmt_dx),
                "correct_diagnosis": ground_truth,
                "correctness": tmt_correct,
                "transcript_text": tmt_consult,
                "run_id": 0,
                "random_seed": self.seed,
                **tmt_meta,
            }

            # Persist treatment transcript for replay/oracle in future epochs.
            transcript_path = os.path.join(transcript_dir, f"case_{case_id}.json")
            with open(transcript_path, "w", encoding="utf-8") as fh:
                json.dump(tmt_record, fh, indent=2, ensure_ascii=False)
            tmt_record["transcript_id"] = transcript_path

            log_deployment_case(tmt_record, self.output_dir)
            new_epoch_treatment_records.append(tmt_record)

            print(
                f"  case {case_id:4d}  "
                f"ctrl={'Y' if ctrl_correct else 'N'}  "
                f"tmt={'Y' if tmt_correct else 'N'}"
            )

        # Advance global pointer by the number of cases actually streamed.
        self._global_start += len(new_epoch_treatment_records)

        # ── Phase 2 + 3: Replay & Oracle on past cases (skip first epoch) ────
        if epoch_idx > 0 and self.past_records:
            # past_records contains only treatment records (one per past case).
            past_n = len(self.past_records)
            current_n = len(new_epoch_treatment_records)
            total_n = past_n + current_n

            print(f"\n[Phase 2] Historical replay + oracle — {past_n} past cases")

            replay_results = []
            replay_metas = []
            oracle_results = []
            oracle_metas = []

            for past_rec in self.past_records:
                # 2a. Historical replay — new treatment reads saved transcript
                replay_dx, replay_correct, replay_meta = run_replay_case(
                    transcript_text=past_rec["transcript_text"],
                    correct_diagnosis=past_rec["correct_diagnosis"],
                    new_model=treatment_model,
                    moderator_llm=self.moderator_llm,
                )
                replay_rec = {
                    **past_rec,
                    "epoch_id": epoch_id,
                    "version_id": version["version_id"],
                    "model_name": treatment_model,
                    "arm": "treatment",
                    "evaluation_type": "historical_replay",
                    "source_epoch": past_rec["epoch_id"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "diagnosis": str(replay_dx),
                    "correctness": replay_correct,
                    "run_id": 1,
                }
                log_deployment_case(replay_rec, self.output_dir)
                replay_results.append(replay_rec)
                replay_metas.append(replay_meta)

                # 2b. Oracle full replay — fresh interactive re-run from scratch
                oracle_dx, oracle_correct, oracle_transcript, oracle_meta = run_oracle_case(
                    dataset=self.dataset,
                    case_id=past_rec["case_id"],
                    new_model=treatment_model,
                    shared_config=self.shared_config,
                )
                oracle_rec = {
                    **past_rec,
                    "epoch_id": epoch_id,
                    "version_id": version["version_id"],
                    "model_name": treatment_model,
                    "arm": "treatment",
                    "evaluation_type": "oracle_full_replay",
                    "source_epoch": past_rec["epoch_id"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "diagnosis": str(oracle_dx),
                    "correctness": oracle_correct,
                    "transcript_text": oracle_transcript,
                    "run_id": 2,
                }
                log_deployment_case(oracle_rec, self.output_dir)
                oracle_results.append(oracle_rec)
                oracle_metas.append(oracle_meta)

                print(
                    f"  past case {past_rec['case_id']:4d}  "
                    f"replay={'Y' if replay_correct else 'N'}  "
                    f"oracle={'Y' if oracle_correct else 'N'}"
                )

            # ── Phase 3: Metrics ─────────────────────────────────────────────
            # All counts are derived directly from records (not from floats).

            # current_correct is shared by both hybrid and oracle denominators.
            current_correct = sum(1 for r in new_epoch_treatment_records if r["correctness"])

            hybrid_past_correct  = sum(1 for r in replay_results  if r["correctness"])
            hybrid_total_correct = hybrid_past_correct + current_correct

            oracle_past_correct  = sum(1 for r in oracle_results  if r["correctness"])
            oracle_total_correct = oracle_past_correct + current_correct

            all_control = self.past_control_records + new_epoch_control_records
            control_correct = sum(1 for r in all_control if r["correctness"])

            def _acc(correct, n):
                return correct / n if n > 0 else None

            def _diff(a, b):
                return (a - b) if (a is not None and b is not None) else None

            # Failure-mode counts: empty_response_n and reasoning_only_response_n.
            # For control/current_treatment: read from records (meta was spread in at log time).
            # For historical_replay/oracle_past: read from separate meta lists so
            # deployment_log.jsonl records remain unchanged.
            def _empty_n(items):
                return sum(1 for i in items if i.get("doctor_empty_response", False))

            def _reasoning_only_n(items):
                return sum(
                    1 for i in items
                    if i.get("doctor_empty_response", False) and i.get("reasoning_content_present", False)
                )

            ctrl_empty_n          = _empty_n(all_control)
            ctrl_reasoning_only_n = _reasoning_only_n(all_control)

            cur_empty_n          = _empty_n(new_epoch_treatment_records)
            cur_reasoning_only_n = _reasoning_only_n(new_epoch_treatment_records)

            rpl_empty_n          = _empty_n(replay_metas)
            rpl_reasoning_only_n = _reasoning_only_n(replay_metas)

            orc_empty_n          = _empty_n(oracle_metas)
            orc_reasoning_only_n = _reasoning_only_n(oracle_metas)

            control_acc = _acc(control_correct,      total_n)
            hybrid_acc  = _acc(hybrid_total_correct,  total_n)
            oracle_acc  = _acc(oracle_total_correct,  total_n)

            print(f"\n[Phase 3] Summary — epoch {epoch_idx}  ({treatment_model})")
            print(f"  total_n:             {total_n}  (past={past_n}  current={current_n})")
            print(f"  Control accuracy:    {control_acc:.4f}  ({control_correct}/{total_n})  empty={ctrl_empty_n}  reasoning_only={ctrl_reasoning_only_n}")
            print(f"  Hybrid estimate:     {hybrid_acc:.4f}  ({hybrid_total_correct}/{total_n})")
            print(f"  Oracle full replay:  {oracle_acc:.4f}  ({oracle_total_correct}/{total_n})")
            print(f"  Hybrid − Oracle:     {_diff(hybrid_acc, oracle_acc):+.4f}")

            summary = {
                "epoch_id": epoch_id,
                "doctor_model_name": self.control_model,
                "patient_llm": self.patient_llm,
                "treatment_model": treatment_model,
                "past_n": past_n,
                "current_n": current_n,
                "total_n": total_n,
                "control": {
                    "correct": control_correct,
                    "n": total_n,
                    "accuracy": control_acc,
                    "empty_response_n": ctrl_empty_n,
                    "reasoning_only_response_n": ctrl_reasoning_only_n,
                },
                "hybrid": {
                    "past_correct": hybrid_past_correct,
                    "current_correct": current_correct,
                    "total_correct": hybrid_total_correct,
                    "n": total_n,
                    "accuracy": hybrid_acc,
                },
                "oracle": {
                    "past_correct": oracle_past_correct,
                    "current_correct": current_correct,
                    "total_correct": oracle_total_correct,
                    "n": total_n,
                    "accuracy": oracle_acc,
                },
                "comparisons": {
                    "hybrid_minus_oracle":  _diff(hybrid_acc,  oracle_acc),
                    "control_minus_hybrid": _diff(control_acc, hybrid_acc),
                    "control_minus_oracle": _diff(control_acc, oracle_acc),
                },
                "components": {
                    "current_treatment": {
                        "correct": current_correct,
                        "n": current_n,
                        "accuracy": _acc(current_correct, current_n),
                        "empty_response_n": cur_empty_n,
                        "reasoning_only_response_n": cur_reasoning_only_n,
                    },
                    "historical_replay": {
                        "correct": hybrid_past_correct,
                        "n": past_n,
                        "accuracy": _acc(hybrid_past_correct, past_n),
                        "empty_response_n": rpl_empty_n,
                        "reasoning_only_response_n": rpl_reasoning_only_n,
                    },
                    "oracle_past": {
                        "correct": oracle_past_correct,
                        "n": past_n,
                        "accuracy": _acc(oracle_past_correct, past_n),
                        "empty_response_n": orc_empty_n,
                        "reasoning_only_response_n": orc_reasoning_only_n,
                    },
                },
            }
            summary_path = os.path.join(self.output_dir, f"{epoch_id}_summary.json")
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
            print(f"  Summary saved: {summary_path}")

        else:
            # Epoch 0: no past cases yet — write a summary from concurrent results only.
            current_n = len(new_epoch_treatment_records)
            current_correct = sum(1 for r in new_epoch_treatment_records if r["correctness"])
            control_correct = sum(1 for r in new_epoch_control_records if r["correctness"])

            def _acc(correct, n):
                return correct / n if n > 0 else None

            def _empty_n(items):
                return sum(1 for i in items if i.get("doctor_empty_response", False))

            def _reasoning_only_n(items):
                return sum(
                    1 for i in items
                    if i.get("doctor_empty_response", False) and i.get("reasoning_content_present", False)
                )

            ctrl_acc = _acc(control_correct, current_n)
            tmt_acc  = _acc(current_correct,  current_n)

            print(f"\n[Phase 3] Epoch 0 summary — ({treatment_model})")
            print(f"  current_n:           {current_n}")
            print(f"  Control accuracy:    {ctrl_acc:.4f}  ({control_correct}/{current_n})")
            print(f"  Treatment accuracy:  {tmt_acc:.4f}  ({current_correct}/{current_n})")

            summary = {
                "epoch_id": epoch_id,
                "doctor_model_name": self.control_model,
                "patient_llm": self.patient_llm,
                "treatment_model": treatment_model,
                "past_n": 0,
                "current_n": current_n,
                "total_n": current_n,
                "note": "Epoch 0: no historical replay or oracle — concurrent paired results only.",
                "control": {
                    "correct": control_correct,
                    "n": current_n,
                    "accuracy": ctrl_acc,
                    "empty_response_n": _empty_n(new_epoch_control_records),
                    "reasoning_only_response_n": _reasoning_only_n(new_epoch_control_records),
                },
                "components": {
                    "current_treatment": {
                        "correct": current_correct,
                        "n": current_n,
                        "accuracy": tmt_acc,
                        "empty_response_n": _empty_n(new_epoch_treatment_records),
                        "reasoning_only_response_n": _reasoning_only_n(new_epoch_treatment_records),
                    },
                },
            }
            summary_path = os.path.join(self.output_dir, f"{epoch_id}_summary.json")
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
            print(f"  Summary saved: {summary_path}")

        # Accumulate for future epochs.
        # past_records: treatment only (for replay/oracle).
        # past_control_records: control only (for control_acc).
        self.past_records.extend(new_epoch_treatment_records)
        self.past_control_records.extend(new_epoch_control_records)
