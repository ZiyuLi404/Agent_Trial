"""Editable measurement-construction surface for HarnessOpt."""


def build_measurement(base_adapter, scenario, config):
    return super(type(base_adapter), base_adapter).build_measurement(scenario, config)
