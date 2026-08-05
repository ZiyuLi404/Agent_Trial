"""Editable interaction-loop surface for HarnessOpt."""

from skill_harness.common.agentclinic.runner import run_case as _baseline_run_case


def run_case(scenario, config, doctor_factory, measurement_factory):
    return _baseline_run_case(
        scenario,
        config,
        doctor_factory=doctor_factory,
        measurement_factory=measurement_factory,
    )
