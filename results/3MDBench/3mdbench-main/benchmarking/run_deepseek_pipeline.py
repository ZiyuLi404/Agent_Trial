import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dialogue_utils import get_patient_prompt


ASSESSMENT_SYSTEM_PROMPT = '''
You are a group of medical experts that assesses a doctor in a dialogue with a patient on the given scale.
Also, you have a photo of the patient's symptom. You don't need to diagnose anything from it; use it only to evaluate the quality of the doctor's work.
You don't need to identify or diagnose the patient. You only need to evaluate the quality of the consultation provided by the doctor.
The scale is given as a JSON dictionary:
{
    "Doctor assessment": {
        "Medical Interviewing Skills": {
            "1.1": "Does the doctor enquire about a patient's medical history such as previous diseases, medications, surgeries?",
            "1.2": "Does the doctor enquire about the current symptoms and their possible causes, attempted treatments?",
            "1.3": "Does the doctor explain the basis of the provided conclusion to the patient?",
        },
        "Humanistic Care": {
            "2.1": "Does the doctor communicate with respect, empathy, and politeness, providing appropriate guidance and avoiding unnecessary extensions?"
            "2.2": "Does the doctor respect individual wishes of the patient?",
        },
        "Comprehensive Diagnostic and Treatment Abilities": {
            "3.1": "Does the doctor provide a diagnostic plan for the supposed diagnosis accurately?",
            "3.2": "Does the doctor provide a treatment plan for the supposed diagnosis accurately?"
        },
        "Overall Clinical Competence": {
            "4.1": "Which level of clinical competence does the doctor demonstrate during the consultation? Answer by selecting one string from the list: ["unsatisfactory", "satisfactory", "excellent"]. Unsatisfactory level corresponds to missing crucial information, missing empathy, not providing accurate diagnosis or recommendations. Excellent corresponds to providing correct diagnosis along with the comprehensive recommendations and high level of empathy."
        }
    }
}

Answer with the dictionary in the same format, but in place of the questions return answers. Each answer should be 0 or 1, where 0 represents "no" and 1 represents "yes" (except the answers for 4.1).
The returned answer should be parsed with the python ast.literal_eval() command with no errors.
'''


DOCTOR_SYSTEM_PROMPT = """
You are a highly experienced doctor with extensive medical expertise in a simulated world.
Your task is to diagnose the patient and provide the recommendations.
The diagnosis should be chosen from this list: ['eczema', 'hives', 'herpes', 'allergic contact dermatitis', 'contact dermatitis', 'seborrheic dermatitis', 'rosacea', 'conjunctivitis', 'tonsillitis', 'chronic lichen', 'keratosis pilaris', 'molluscum contagiosum', 'lichen planus', 'periodontitis', 'caries', 'psoriasis', 'ingrown nail', 'stye', 'acne', 'onychomycosis', 'seborrheic keratosis', 'chalazion', 'vitiligo', 'shingles', 'mycosis', 'actinic keratosis', 'onycholysis', 'chickenpox', 'dental calculus', 'warts', 'stomatitis', 'abscess', 'gingivitis', 'nail dystrophy']
Pay attention to the provided image and use it to make your decisions. Mention the info obtained from the image in the dialogue.

In a conversation, you need to provide a single diagnosis. If you do not have sufficient information yet, then inquire this information from the patient. Ask only one question at a time.
"""


TEXT_ONLY_DOCTOR_SYSTEM_PROMPT = """
You are a highly experienced doctor with extensive medical expertise in a simulated world.
Your task is to diagnose the patient and provide the recommendations.
The diagnosis should be chosen from this list: ['eczema', 'hives', 'herpes', 'allergic contact dermatitis', 'contact dermatitis', 'seborrheic dermatitis', 'rosacea', 'conjunctivitis', 'tonsillitis', 'chronic lichen', 'keratosis pilaris', 'molluscum contagiosum', 'lichen planus', 'periodontitis', 'caries', 'psoriasis', 'ingrown nail', 'stye', 'acne', 'onychomycosis', 'seborrheic keratosis', 'chalazion', 'vitiligo', 'shingles', 'mycosis', 'actinic keratosis', 'onycholysis', 'chickenpox', 'dental calculus', 'warts', 'stomatitis', 'abscess', 'gingivitis', 'nail dystrophy']

In a conversation, you need to provide a single diagnosis. If you do not have sufficient information yet, then inquire this information from the patient. Ask only one question at a time.
"""


DIAG_EXTRACTION_SYSTEM_PROMPT = """
You are a text analysis engine that processes doctor-patient consultation transcripts. Your task is to identify and extract the final diagnosis that the doctor has decided to assign to the patient. Follow these instructions carefully:

1. Identify the Relevant Sentence:
   - Search the entire transcript for the sentence in which the doctor explicitly communicates the final diagnosis.
   - Note that doctors can express diagnoses in many different ways; it does not have to be in the form "your diagnosis is...". Look for alternative phrasing, searching for other wording that indicates a definitive conclusion.
   - Only extract the sentence if you are confident it contains the final diagnosis, not merely a provisional or hypothetical opinion.

2. Extract the Diagnosis:
   - From the identified sentence, extract the diagnosis. If you are sure that in this sentence, the doctor mentioned multiple diagnoses with an equal confidence level, extract all diagnoses.
   - Ensure that the diagnoses you extract are the ones the doctor confirms as final.
   - Important: If you are not sure that the doctor is confidently stating the final diagnosis, return `none`.

3. Output Format:
   - Provide the extracted diagnosis or diagnoses as a comma-separated list.
   - Do not include any additional text, context, or commentary in your output.
"""


class DeepSeekChatAgent:
    def __init__(self, client, model, system_prompt, max_tokens, temperature=0.6, top_p=0.9):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.history = [{"role": "system", "content": system_prompt}]

    def run(self, content):
        self.history.append({"role": "user", "content": content})
        answer = create_chat_completion(
            self.client,
            self.model,
            self.history,
            self.max_tokens,
            self.temperature,
            self.top_p,
        )
        self.history.append({"role": "assistant", "content": answer})
        return answer


def create_chat_completion(client, model, messages, max_tokens, temperature, top_p):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content or ""


def image_to_data_url(img):
    x = 400
    perc = x / img.size[0]
    img = img.resize((int(img.size[0] * perc), int(img.size[1] * perc)))
    img = img.convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def doctor_content(patient_utterance, img=None):
    if img is None:
        return patient_utterance

    content = [{"type": "text", "text": patient_utterance}]
    content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})
    return content


def get_doctor_replicas(dialogue):
    utterances = [x.strip().lower() for x in re.split("Patient:|Doctor:|DIAG:", dialogue) if x.strip()]
    return [utterances[i] for i in range(1, len(utterances), 2)]


def run_dialogue(patient_client, doctor_client, args, case):
    patient_prompt = get_patient_prompt(
        case["general_complaint"],
        case["complaints"],
        case["personality"],
    )
    patient_agent = DeepSeekChatAgent(
        patient_client,
        args.patient_model,
        patient_prompt,
        args.max_tokens,
        temperature=args.patient_temperature,
        top_p=args.top_p,
    )
    doctor_agent = DeepSeekChatAgent(
        doctor_client,
        args.doctor_model,
        DOCTOR_SYSTEM_PROMPT if args.include_image else TEXT_ONLY_DOCTOR_SYSTEM_PROMPT,
        args.max_tokens,
        temperature=args.doctor_temperature,
        top_p=args.top_p,
    )

    result = ""
    patient_utterance = patient_agent.run("Start the consultation now with only your main complaint.")
    result += f"Patient: {patient_utterance}\n"

    for turn_idx in range(args.max_turns):
        img = case["image"] if turn_idx == 0 and args.include_image else None
        doc_utterance = doctor_agent.run(doctor_content(patient_utterance, img))
        result += f"Doctor: {doc_utterance}\n"

        patient_utterance = patient_agent.run(doc_utterance)
        result += f"Patient: {patient_utterance}\n"

        if "BREAK" in patient_utterance:
            return result, True

    return result, False


def run_assessment(client, args, case, dialogue):
    assessment_prompt = ASSESSMENT_SYSTEM_PROMPT
    if not args.include_image:
        assessment_prompt = assessment_prompt.replace(
            "Also, you have a photo of the patient's symptom. You don't need to diagnose anything from it; use it only to evaluate the quality of the doctor's work.",
            "No image is provided; evaluate only the text dialogue.",
        )
    req = f"""{assessment_prompt}
Ground truth diagnosis: {case['diagnosis']}
Dialogue between the patient and the doctor: {dialogue.replace('BREAK', '/nEnd of the dialogue. Assessment:')}
"""
    if not args.include_image:
        content = req
        return create_chat_completion(
            client,
            args.assessment_model,
            [{"role": "user", "content": content}],
            args.max_tokens,
            args.assessment_temperature,
            args.top_p,
        )

    content = [{"type": "text", "text": req}]
    content.append({"type": "image_url", "image_url": {"url": image_to_data_url(case["image"])}})
    return create_chat_completion(
        client,
        args.assessment_model,
        [{"role": "user", "content": content}],
        args.max_tokens,
        args.assessment_temperature,
        args.top_p,
    )


def run_diag_extraction(client, args, dialogue):
    doctor_replicas = " ".join(get_doctor_replicas(dialogue))
    req = f"{DIAG_EXTRACTION_SYSTEM_PROMPT}\n\nDoctor's replicas: {doctor_replicas}"
    return create_chat_completion(
        client,
        args.diagnosis_model,
        [{"role": "user", "content": req}],
        args.max_tokens,
        args.assessment_temperature,
        args.top_p,
    )


def parse_case_ids(raw):
    if "-" in raw:
        start, end = raw.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def iter_selected_cases(dataset, case_ids, streaming):
    if not streaming:
        for case_id in case_ids:
            yield case_id, dataset[case_id]
        return

    wanted = set(case_ids)
    max_case_id = max(wanted)
    for case_id, case in enumerate(dataset):
        if case_id in wanted:
            yield case_id, case
        if case_id >= max_case_id:
            break


def main():
    parser = argparse.ArgumentParser(description="Run 3MDBench with DeepSeek-compatible chat models.")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--case_ids", required=True, help="Case ids, e.g. 0-9 or 0,2,4.")
    parser.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base_url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--doctor_api_key_env", default=None)
    parser.add_argument("--doctor_base_url", default=None)
    parser.add_argument("--patient_model", default="deepseek-v4-flash")
    parser.add_argument("--doctor_model", default="deepseek-v4-pro")
    parser.add_argument("--assessment_model", default="deepseek-v4-pro")
    parser.add_argument("--diagnosis_model", default="deepseek-v4-pro")
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--patient_temperature", type=float, default=0.6)
    parser.add_argument("--doctor_temperature", type=float, default=0.6)
    parser.add_argument("--assessment_temperature", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no_streaming", action="store_true")
    parser.add_argument("--no_image", action="store_true")
    args = parser.parse_args()
    args.include_image = not args.no_image
    args.streaming = not args.no_streaming

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable {args.api_key_env}.")

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)
    if args.doctor_api_key_env or args.doctor_base_url:
        doctor_api_key_env = args.doctor_api_key_env or args.api_key_env
        doctor_api_key = os.environ.get(doctor_api_key_env)
        if not doctor_api_key:
            raise RuntimeError(f"Missing doctor API key in environment variable {doctor_api_key_env}.")
        doctor_client = OpenAI(
            api_key=doctor_api_key,
            base_url=args.doctor_base_url or args.base_url,
            timeout=args.timeout,
        )
    else:
        doctor_client = client

    print(f"Loading dataset univanxx/3mdbench split=test streaming={args.streaming}", flush=True)
    dataset = load_dataset("univanxx/3mdbench", split="test", streaming=args.streaming)
    case_ids = parse_case_ids(args.case_ids)

    for case_id, case in tqdm(iter_selected_cases(dataset, case_ids, args.streaming), total=len(case_ids)):
        case_key = str(case_id)
        dialogue_path = PROJECT_ROOT / "results" / args.experiment_name / f"case_{case_key}.json"
        assessment_path = PROJECT_ROOT / "results" / "assessment" / args.experiment_name / f"case_{case_key}.json"
        diag_path = PROJECT_ROOT / "results" / "assessment" / "diags" / args.experiment_name / f"case_{case_key}.json"

        try:
            print(f"[case {case_key}] dialogue start", flush=True)
            dialogue, dialogue_ended = run_dialogue(client, doctor_client, args, case)
            dialogue_result = {
                case_key: {
                    "dialogue": dialogue,
                    "dialogue_ended": dialogue_ended,
                    "diagnosis": case["diagnosis"].lower(),
                }
            }
            write_json(dialogue_path, dialogue_result)
            print(f"[case {case_key}] dialogue done ended={dialogue_ended}", flush=True)

            if not dialogue_ended:
                write_json(assessment_path, {case_key: {"assessment": "dialogue_unfinished"}})
                write_json(diag_path, {case_key: {"assessment": "dialogue_unfinished"}})
                continue

            print(f"[case {case_key}] assessment start", flush=True)
            assessment = run_assessment(client, args, case, dialogue)
            write_json(assessment_path, {case_key: {"assessment": assessment}})
            print(f"[case {case_key}] assessment done", flush=True)

            print(f"[case {case_key}] diagnosis extraction start", flush=True)
            diags = run_diag_extraction(client, args, dialogue)
            write_json(diag_path, {case_key: {"diags": diags}})
            print(f"[case {case_key}] diagnosis extraction done", flush=True)
        except Exception as exc:
            error = {"error": str(exc)}
            write_json(dialogue_path, {case_key: error})
            write_json(assessment_path, {case_key: error})
            write_json(diag_path, {case_key: error})
            print(f"failed for {case_key} with error {exc}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    if doctor_client is not client:
        doctor_client.close()
    client.close()


if __name__ == "__main__":
    main()
