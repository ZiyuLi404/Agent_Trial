"""Editable doctor-construction surface for HarnessOpt."""


def build_doctor(base_adapter, scenario, config):
    return super(type(base_adapter), base_adapter).build_doctor(scenario, config)
