from AgentClinic.agentclinic import DoctorAgent
from agent_system_adapters.agentclinic.skill_doctor import SkillDoctorAgent
from change_generators.skills import SkillArtifact
from trial.trial_manager import run_case


class AgentClinicAdapter:
    """Apply an optional skill artifact without changing AgentClinic core code."""

    def __init__(self, skill_artifact: SkillArtifact | None = None):
        self.skill_artifact = skill_artifact

    @property
    def condition_name(self) -> str:
        return "with_skill" if self.skill_artifact is not None else "no_skill"

    def build_doctor(self, scenario, config):
        kwargs = {
            "scenario": scenario,
            "backend_str": config["doctor_llm"],
            "max_infs": config.get("total_inferences", 20),
        }
        if self.skill_artifact is None:
            return DoctorAgent(**kwargs)
        return SkillDoctorAgent(**kwargs, skill_artifact=self.skill_artifact)

    def evaluate_case(self, scenario, config):
        result = run_case(scenario, config, doctor_factory=self.build_doctor)
        diagnosis, correctness, dialogue, meta = result
        meta = {
            **meta,
            "variant": self.variant_metadata(),
        }
        return diagnosis, correctness, dialogue, meta

    def variant_metadata(self) -> dict:
        if self.skill_artifact is None:
            return {"condition": "no_skill", "skill": None}
        return {
            "condition": "with_skill",
            "skill": self.skill_artifact.to_dict(include_content=False),
        }
