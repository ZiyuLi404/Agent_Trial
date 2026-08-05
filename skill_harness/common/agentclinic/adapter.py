from AgentClinic.agentclinic import DoctorAgent, MeasurementAgent, load_doctor_prompt_template
from skill_harness.artifacts import HarnessArtifact, SkillArtifact
from skill_harness.common.agentclinic.grounded_measurement import GroundedMeasurementAgent
from skill_harness.common.agentclinic.runner import run_case
from skill_harness.common.agentclinic.skill_doctor import SkillDoctorAgent


class AgentClinicAdapter:
    """Apply optional skill/harness artifacts without changing AgentClinic core."""

    def __init__(
        self,
        skill_artifact: SkillArtifact | None = None,
        harness_artifact: HarnessArtifact | None = None,
    ):
        self.skill_artifact = skill_artifact
        self.harness_artifact = harness_artifact
        if harness_artifact is not None and harness_artifact.agent_system != "agentclinic":
            raise ValueError(
                f"Harness targets {harness_artifact.agent_system}, not agentclinic"
            )

    @property
    def condition_name(self) -> str:
        if self.skill_artifact is None and self.harness_artifact is None:
            return "baseline"
        return "variant"

    def effective_config(self, config) -> dict:
        effective = dict(config)
        if self.harness_artifact is not None:
            max_inferences = self.harness_artifact.doctor_config.get("max_inferences")
            if max_inferences is not None:
                if not isinstance(max_inferences, int) or max_inferences <= 0:
                    raise ValueError("Harness doctor.max_inferences must be a positive integer")
                effective["total_inferences"] = max_inferences
        return effective

    def build_doctor(self, scenario, config):
        config = self.effective_config(config)
        kwargs = {
            "scenario": scenario,
            "backend_str": config["doctor_llm"],
            "max_infs": config.get("total_inferences", 20),
        }
        if self.harness_artifact is not None:
            doctor_config = self.harness_artifact.doctor_config
            prompt_style = doctor_config.get("prompt_style")
            if prompt_style:
                prompt_bank = doctor_config.get("prompt_bank", "doctor_prompts.json")
                kwargs["doctor_prompt_template"] = load_doctor_prompt_template(
                    prompt_bank, prompt_style
                )
        if self.skill_artifact is None:
            return DoctorAgent(**kwargs)
        return SkillDoctorAgent(**kwargs, skill_artifact=self.skill_artifact)

    def build_measurement(self, scenario, config):
        measurement_config = (
            self.harness_artifact.measurement_config
            if self.harness_artifact is not None else {}
        )
        mode = measurement_config.get("mode", "generative")
        if mode == "generative":
            return MeasurementAgent(
                scenario=scenario,
                backend_str=config["measurement_llm"],
            )
        if mode == "grounded_lookup":
            return GroundedMeasurementAgent(
                scenario=scenario,
                missing_policy=measurement_config.get("missing_policy", "unavailable"),
            )
        raise ValueError(f"Unknown measurement mode: {mode!r}")

    def evaluate_case(self, scenario, config):
        effective_config = self.effective_config(config)
        result = run_case(
            scenario,
            effective_config,
            doctor_factory=self.build_doctor,
            measurement_factory=self.build_measurement,
        )
        diagnosis, correctness, dialogue, meta = result
        meta = {
            **meta,
            "variant": self.variant_metadata(),
        }
        return diagnosis, correctness, dialogue, meta

    def variant_metadata(self) -> dict:
        return {
            "condition": self.condition_name,
            "skill": (
                self.skill_artifact.to_dict(include_content=False)
                if self.skill_artifact is not None else None
            ),
            "harness": (
                self.harness_artifact.to_dict(include_config=True)
                if self.harness_artifact is not None else None
            ),
        }
