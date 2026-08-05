"""Baseline HarnessOpt workspace. The optimizer may edit this package only."""

from skill_harness.common.agentclinic import AgentClinicAdapter
from skill_harness.methods.harnessopt.workspace.agentclinic.doctor import build_doctor
from skill_harness.methods.harnessopt.workspace.agentclinic.measurement import (
    build_measurement,
)
from skill_harness.methods.harnessopt.workspace.agentclinic.runner import run_case


class HarnessOptAgentClinicAdapter(AgentClinicAdapter):
    def build_doctor(self, scenario, config):
        return build_doctor(self, scenario, config)

    def build_measurement(self, scenario, config):
        return build_measurement(self, scenario, config)

    def evaluate_case(self, scenario, config):
        effective = self.effective_config(config)
        diagnosis, correctness, dialogue, meta = run_case(
            scenario,
            effective,
            doctor_factory=self.build_doctor,
            measurement_factory=self.build_measurement,
        )
        return diagnosis, correctness, dialogue, {
            **meta,
            "variant": self.variant_metadata(),
            "optimizer": "harnessopt",
        }
