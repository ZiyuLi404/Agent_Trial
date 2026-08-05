from skill_harness.artifacts import SkillArtifact
from skill_harness.common.agentclinic import AgentClinicAdapter, LOADERS
from skill_harness.methods.harnessopt.workspace.agentclinic import (
    HarnessOptAgentClinicAdapter,
)


def test_round_zero_harness_matches_common_adapter_prompt():
    skill = SkillArtifact.load(
        "skill_harness/artifacts/seeds/diagnostic_reasoning/v000.md"
    )
    scenario = LOADERS["MedQA"]().get_scenario(id=0)
    config = {"doctor_llm": "deepseek-v4-pro", "total_inferences": 8}
    baseline = AgentClinicAdapter(skill_artifact=skill)
    harnessopt = HarnessOptAgentClinicAdapter(skill_artifact=skill)
    assert harnessopt.build_doctor(scenario, config).system_prompt() == (
        baseline.build_doctor(scenario, config).system_prompt()
    )
