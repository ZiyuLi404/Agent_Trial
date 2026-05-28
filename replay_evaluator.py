"""
Historical-replay evaluator for deployment_replay mode.

The new treatment model reads a saved past transcript and provides a diagnosis
without re-interacting with the patient.  This mirrors what is practically
available in a real clinic — you can replay archived notes but cannot
call patients back.

Correctness is judged by the same moderator LLM used in all other modes.
"""

from AgentClinic.agentclinic import query_model, normalize_answer, compare_results


_REPLAY_SYSTEM = (
    "You are an expert diagnostician reviewing a past doctor-patient consultation. "
    "Read the transcript carefully and provide your best diagnosis based solely on "
    "the information visible in the transcript. "
    "Respond exactly with: DIAGNOSIS READY: <single most likely diagnosis>"
)


def run_replay_case(transcript_text, correct_diagnosis, new_model, moderator_llm):
    """Read a saved transcript and return (raw_diagnosis_str, correctness_bool).

    No patient interaction occurs — the model diagnoses from the transcript alone.

    Parameters
    ----------
    transcript_text : str
        Full doctor/patient/measurement dialogue from the original run.
    correct_diagnosis : str
        Ground-truth diagnosis string for the moderator to compare against.
    new_model : str
        Backend model string for the new treatment being evaluated.
    moderator_llm : str
        Backend model string for the correctness-judging moderator.

    Returns
    -------
    (str, bool)
        Raw model output and whether the moderator judged it correct.
    """
    prompt = (
        "Here is a recorded doctor-patient consultation transcript:\n\n"
        + str(transcript_text)
        + "\n\nBased on this consultation, provide your diagnosis.\n"
        "Format: DIAGNOSIS READY: <single most likely diagnosis>"
    )

    raw_answer = query_model(new_model, prompt, _REPLAY_SYSTEM)
    result = compare_results(raw_answer, str(correct_diagnosis), moderator_llm, None)
    correctness = normalize_answer(result).lower().startswith("yes")
    return raw_answer, correctness
