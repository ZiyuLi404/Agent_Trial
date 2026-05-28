"""
Oracle full-replay evaluator for deployment_replay mode.

The new treatment model re-interacts from scratch with the patient agent for
every past case, giving the counterfactual ground truth: what accuracy *would*
this model have achieved if it had been deployed from the start?

In a real clinic this is impossible (patients don't come back), but in
simulation it serves as the reference that the hybrid estimate is measured
against.

Reuses trial_manager.run_case() so the interaction loop is identical to
every other eval mode.
"""

from trial_manager import run_case, LOADERS


def run_oracle_case(dataset, case_id, new_model, shared_config):
    """Re-run a past case interactively with the new model from scratch.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g. "MedQA") — used to reload the original scenario.
    case_id : int
        Global case ID as stored in the transcript record.  The scenario is
        re-loaded as ``case_id % loader.num_scenarios`` to handle wrapping.
    new_model : str
        Backend model string for the new treatment being evaluated.
    shared_config : dict
        Must contain patient_llm, measurement_llm, moderator_llm,
        total_inferences.  doctor_llm is overridden with new_model.

    Returns
    -------
    (str, bool, str)
        diagnosis, correctness, full consultation transcript.
    """
    loader = LOADERS[dataset]()
    scenario = loader.get_scenario(id=case_id % loader.num_scenarios)
    config = {**shared_config, "doctor_llm": new_model}
    diagnosis, correctness, consultation = run_case(scenario, config)
    return diagnosis, correctness, consultation
