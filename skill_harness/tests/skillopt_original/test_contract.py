import sys

from skill_harness.upstream import verify_checkout


def _load_original_modules():
    upstream = verify_checkout("SkillOpt")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from skill_harness.methods.skillopt_original.dataloader import (
        AgentClinicDataLoader,
    )
    from skill_harness.methods.skillopt_original.rollout import run_batch

    return AgentClinicDataLoader, run_batch


def test_manifest_contract_and_parallel_dry_rollout(tmp_path):
    dataloader_cls, run_batch = _load_original_modules()
    dataloader = dataloader_cls(
        "skill_harness/experiments/agentclinic/manifests/medqa_pure_v1.json"
    )
    dataloader.setup({})
    items = dataloader.build_train_batch(2, seed=7).payload
    results = run_batch(
        items=items,
        skill_content="# Test skill\nAsk focused questions.",
        out_root=str(tmp_path),
        agentclinic_repo=".",
        doctor_llm="deepseek-v4-pro",
        patient_llm="deepseek-v4-flash",
        measurement_llm="deepseek-v4-flash",
        moderator_llm="deepseek-v4-flash",
        total_inferences=2,
        contract_dry_run=True,
        workers=2,
    )
    assert len(results) == 2
    assert all(row["hard"] == 0 for row in results)
    assert tmp_path.joinpath("metrics.json").is_file()
