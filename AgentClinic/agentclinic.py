import argparse
import os
import re
import random
import time
import json
from pathlib import Path

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import replicate
except ImportError:
    replicate = None

from openai import OpenAI

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

llama2_url = "meta/llama-2-70b-chat"
llama3_url = "meta/meta-llama-3-70b-instruct"
mixtral_url = "mistralai/mixtral-8x7b-instruct-v0.1"

OPENAI_MODELS = {
    "gpt4": "gpt-4-turbo-preview",
    "gpt4v": "gpt-4-vision-preview",
    "gpt3.5": "gpt-3.5-turbo",
    "gpt4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "o1-preview": "o1-preview-2024-09-12",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-5.4-nano": "gpt-5.4-nano",
    "gpt-5.5": "gpt-5.5",
}

# GPT-5.x are reasoning models: they reject custom `temperature`, require
# `max_completion_tokens` instead of `max_tokens`, and accept `reasoning_effort`.
# Reasoning tokens are billed against the completion budget, so the limit must be
# generous or the visible answer can come back empty.
OPENAI_REASONING_MODELS = {
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.5",
}

DEEPSEEK_MODELS = {
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
}


def resolve_local_path(path_str: str) -> Path:
    """Resolve a path relative to this file's directory if not absolute."""
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def load_doctor_prompt_template(prompt_json_path: str, prompt_style: str) -> dict:
    path = resolve_local_path(prompt_json_path)

    if not path.exists():
        raise FileNotFoundError(f"Doctor prompt JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        text = "\n".join(line for line in f if not line.lstrip().startswith("//"))
    prompt_bank = json.loads(text)

    if prompt_style not in prompt_bank:
        available = ", ".join(prompt_bank.keys())
        raise ValueError(
            f"Unknown doctor prompt style: '{prompt_style}'. "
            f"Available styles: {available}"
        )

    return prompt_bank[prompt_style]


def normalize_icd10cm_code(code: str) -> str:
    """Strip dots and uppercase so 'L72.0', 'l72.0', and 'L720' all map to 'L720'."""
    if not code:
        return ""
    return re.sub(r"[^A-Z0-9]", "", code.upper().strip())


def load_icd10cm_dictionary(jsonl_path: str) -> dict:
    """Load ICD-10-CM JSONL into a dict keyed by normalized code.

    Returns: {"L720": {"code": "L720", "description": "Epidermal cyst"}, ...}
    """
    path = resolve_local_path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"ICD-10-CM JSONL file not found: {path}")

    code_dict = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}")
            code = record.get("code", "").strip()
            if not code:
                continue
            code_dict[normalize_icd10cm_code(code)] = record

    return code_dict


def parse_icd10cm_from_diagnosis(raw_output: str) -> dict:
    """Extract ICD-10-CM code and label from a doctor diagnosis string."""
    text = (raw_output or "").strip()

    code_match = re.search(
        r"ICD10CM_CODE:\s*([A-Z][0-9][A-Z0-9](?:\.?[A-Z0-9]{0,4})?)",
        text,
        flags=re.IGNORECASE,
    )
    label_match = re.search(
        r"ICD10CM_LABEL:\s*([^;\n\r]+)",
        text,
        flags=re.IGNORECASE,
    )

    code = code_match.group(1).upper().strip() if code_match else ""
    label = label_match.group(1).strip() if label_match else ""

    return {
        "icd10cm_code": code,
        "icd10cm_code_normalized": normalize_icd10cm_code(code),
        "icd10cm_label": label,
        "raw_final_diagnosis": text,
    }


def validate_icd10cm_output(parsed: dict, icd10cm_dict: dict) -> dict:
    """Add validation fields to a parsed ICD-10-CM record."""
    normalized = parsed.get("icd10cm_code_normalized", "")
    match = icd10cm_dict.get(normalized)

    if match is None:
        parsed["icd10cm_valid"] = False
        parsed["icd10cm_dictionary_label"] = ""
        parsed["icd10cm_dictionary_code"] = ""
    else:
        parsed["icd10cm_valid"] = True
        parsed["icd10cm_dictionary_label"] = (
            match.get("description")
            or match.get("long_description")
            or match.get("short_description")
            or ""
        )
        parsed["icd10cm_dictionary_code"] = match.get("code", "")

    return parsed


def get_diagnosis_bucket_key(record: dict) -> str:
    """Return the canonical string used as the bucket key for a diagnosis record."""
    if record.get("diagnosis_output_format") == "icd10cm":
        if record.get("icd10cm_valid") and record.get("icd10cm_dictionary_code"):
            return record["icd10cm_dictionary_code"]
        if record.get("icd10cm_code"):
            return record["icd10cm_code"]
        return record.get("raw_final_diagnosis", "").strip()
    return record.get("final_diagnosis", "").strip()


def load_huggingface_model(model_name):
    if pipeline is None:
        raise ImportError("transformers is not installed. Run: pip install transformers")
    pipe = pipeline("text-generation", model=model_name, device_map="auto")
    return pipe


def inference_huggingface(prompt, pipe):
    response = pipe(prompt, max_new_tokens=1000)[0]["generated_text"]
    response = response.replace(prompt, "")
    return response


def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Pass --openai_api_key or set it as an environment variable.")
    return OpenAI(api_key=api_key)


def get_deepseek_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set. Pass --deepseek_api_key or set it as an environment variable.")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def normalize_answer(answer):
    if answer is None:
        return ""
    return re.sub(r"\s+", " ", str(answer)).strip()


def query_model(
    model_str,
    prompt,
    system_prompt,
    tries=30,
    timeout=20.0,
    image_requested=False,
    scene=None,
    max_prompt_len=2**14,
    clip_prompt=False,
):
    supported_models = list(OPENAI_MODELS.keys()) + list(DEEPSEEK_MODELS) + [
        "claude3.5sonnet",
        "llama-2-70b-chat",
        "mixtral-8x7b",
        "llama-3-70b-instruct",
    ]
    if model_str not in supported_models and "_HF" not in model_str:
        raise Exception("No model by the name {}".format(model_str))

    if clip_prompt:
        prompt = prompt[:max_prompt_len]

    last_error = None
    for _ in range(tries):
        try:
            # -------------------------
            # DeepSeek API
            # -------------------------
            if model_str in DEEPSEEK_MODELS:
                if image_requested:
                    raise Exception(
                        "DeepSeek text API does not support image input in this script. "
                        "Use MedQA / MedQA_Ext, or disable --doctor_image_request for DeepSeek."
                    )

                client = get_deepseek_client()
                response = client.chat.completions.create(
                    model=model_str,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.05,
                    max_tokens=2000,
                    stream=False,
                )
                return normalize_answer(response.choices[0].message.content)

            # -------------------------
            # OpenAI API, new SDK style
            # -------------------------
            if model_str in OPENAI_MODELS:
                client = get_openai_client()
                api_model = OPENAI_MODELS[model_str]

                if image_requested:
                    if scene is None or not hasattr(scene, "image_url"):
                        raise Exception("image_requested=True, but this scenario has no image_url.")
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": str(scene.image_url)}},
                            ],
                        },
                    ]
                elif model_str == "o1-preview":
                    messages = [{"role": "user", "content": system_prompt + "\n" + prompt}]
                    response = client.chat.completions.create(
                        model=api_model,
                        messages=messages,
                    )
                    return normalize_answer(response.choices[0].message.content)
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]

                if model_str in OPENAI_REASONING_MODELS:
                    # No max_completion_tokens => model default (effectively
                    # unbounded up to context), so reasoning tokens never starve
                    # the visible answer. temperature is unsupported here.
                    response = client.chat.completions.create(
                        model=api_model,
                        messages=messages,
                        reasoning_effort="medium",
                    )
                else:
                    response = client.chat.completions.create(
                        model=api_model,
                        messages=messages,
                        temperature=0.05,
                        max_tokens=2000,
                    )
                return normalize_answer(response.choices[0].message.content)

            # -------------------------
            # Anthropic
            # -------------------------
            if model_str == "claude3.5sonnet":
                if anthropic is None:
                    raise ImportError("anthropic is not installed. Run: pip install anthropic")
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    system=system_prompt,
                    max_tokens=2560,
                    messages=[{"role": "user", "content": prompt}],
                )
                answer = json.loads(message.to_json())["content"][0]["text"]
                return normalize_answer(answer)

            # -------------------------
            # Replicate models
            # -------------------------
            if model_str == "llama-2-70b-chat":
                if replicate is None:
                    raise ImportError("replicate is not installed. Run: pip install replicate")
                output = replicate.run(
                    llama2_url,
                    input={"prompt": prompt, "system_prompt": system_prompt, "max_new_tokens": 2000},
                )
                return normalize_answer("".join(output))

            if model_str == "mixtral-8x7b":
                if replicate is None:
                    raise ImportError("replicate is not installed. Run: pip install replicate")
                output = replicate.run(
                    mixtral_url,
                    input={"prompt": prompt, "system_prompt": system_prompt, "max_new_tokens": 750},
                )
                return normalize_answer("".join(output))

            if model_str == "llama-3-70b-instruct":
                if replicate is None:
                    raise ImportError("replicate is not installed. Run: pip install replicate")
                output = replicate.run(
                    llama3_url,
                    input={"prompt": prompt, "system_prompt": system_prompt, "max_new_tokens": 2000},
                )
                return normalize_answer("".join(output))

            if "HF_" in model_str:
                raise Exception("HuggingFace inference is not implemented in this script.")

        except Exception as e:
            last_error = e
            time.sleep(timeout)
            continue

    raise Exception("Max retries: timeout. Last error: {}".format(last_error))


class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQA:
    def __init__(self) -> None:
        with open("AgentClinic/agentclinic_medqa.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
        


class ScenarioMedQAExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMedQAExtended:
    def __init__(self) -> None:
        with open("AgentClinic/agentclinic_medqa_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQAExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]
        


class ScenarioMIMICIVQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info  = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info  = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]
    
    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info
    
    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams
    
    def diagnosis_information(self) -> dict:
        return self.diagnosis


class ScenarioLoaderMIMICIV:
    def __init__(self) -> None:
        with open("AgentClinic/agentclinic_mimiciv.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMIMICIVQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJMExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict 
        self.question = scenario_dict["question"] 
        self.image_url = scenario_dict["image_url"] 
        self.diagnosis = [_sd["text"] 
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
    
    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams
    
    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJMExtended:
    def __init__(self) -> None:
        with open("AgentClinic/agentclinic_nejm_extended.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJMExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class ScenarioNEJM:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict 
        self.question = scenario_dict["question"] 
        self.image_url = scenario_dict["image_url"] 
        self.diagnosis = [_sd["text"] 
            for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"
    
    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams
    
    def diagnosis_information(self) -> str:
        return self.diagnosis


class ScenarioLoaderNEJM:
    def __init__(self) -> None:
        with open("AgentClinic/agentclinic_nejm.jsonl", "r") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJM(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)
    
    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]
    
    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]


class PatientAgent:
    def __init__(self, scenario, backend_str="gpt4", bias_present=None) -> None:
        # disease of patient, or "correct answer"
        self.disease = ""
        # symptoms that patient presents
        self.symptoms = ""
        # conversation history between doctor and patient
        self.agent_hist = ""
        # language model backend for patient agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # sample initial question from dataset
        self.scenario = scenario
        self.reset()
        self.pipe = None

        self.biases = ["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "self_diagnosis":
            return "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_patient(self, question) -> str:
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the doctor response: " + question + "Now please continue your dialogue\nPatient: ", self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."""
        symptoms = "\n\nBelow is all of your information. {}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked.".format(self.symptoms)
        return base + bias_prompt + symptoms
    
    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"


class DoctorAgent:
    def __init__(self, scenario, backend_str="gpt4", max_infs=20, bias_present=None, img_request=False, doctor_prompt_template=None) -> None:
        # number of inference calls to the doctor
        self.infs = 0
        # maximum number of inference calls to the doctor
        self.MAX_INFS = max_infs
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for doctor
        self.presentation = ""
        # language model backend for doctor agent
        self.backend = backend_str
        # presentation of any form of bias
        self.bias_present = (None if bias_present == "None" else bias_present)
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.reset()
        self.pipe = None
        self.img_request = img_request
        self.doctor_prompt_template = doctor_prompt_template
        self.biases = ["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"]

    def generate_bias(self) -> str:
        """ 
        ================
        Cognitive biases 
        ================
        """
        if self.bias_present == "recency":
            return "\nRecently, you learned that a doctor with similar symptoms to the patient that diagnosed their patient with cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "status_quo":
            return "\nYour hosptial has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n"
        elif self.bias_present == "confirmation":
            return "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_doctor(self, question, image_requested=False) -> str:
        answer = str()
        if self.infs >= self.MAX_INFS: return "Maximum inferences reached"
        answer = query_model(self.backend, "\nHere is a history of your dialogue: " + self.agent_hist + "\n Here was the patient response: " + question + "Now please continue your dialogue\nDoctor: ", self.system_prompt(), image_requested=image_requested, scene=self.scenario)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()

        if self.doctor_prompt_template is None:
            base = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {} questions total before you must make a decision. You have asked {} questions so far. You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\"".format(self.MAX_INFS, self.infs) + ("You may also request medical images related to the disease to be returned with \"REQUEST IMAGES\"." if self.img_request else "")
            presentation_section = "\n\nBelow is all of the information you have. {}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams.".format(self.presentation)
            return base + bias_prompt + presentation_section

        template = self.doctor_prompt_template
        fmt = dict(max_infs=self.MAX_INFS, infs=self.infs, presentation=self.presentation)
        base = template["base_prompt"].format(**fmt)
        image_prompt = template.get("image_prompt", "") if self.img_request else ""
        presentation_section = template["presentation_prompt"].format(**fmt)
        return base + image_prompt + bias_prompt + presentation_section

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()


class MeasurementAgent:
    def __init__(self, scenario, backend_str="gpt4") -> None:
        # conversation history between doctor and patient
        self.agent_hist = ""
        # presentation information for measurement 
        self.presentation = ""
        # language model backend for measurement agent
        self.backend = backend_str
        # prepare initial conditions for LLM
        self.scenario = scenario
        self.pipe = None
        self.reset()

    def inference_measurement(self, question) -> str:
        answer = str()
        answer = query_model(self.backend, "\nHere is a history of the dialogue: " + self.agent_hist + "\n Here was the doctor measurement request: " + question, self.system_prompt())
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def system_prompt(self) -> str:
        base = "You are an measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(self.information)
        return base + presentation
    
    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()


def compare_results(diagnosis, correct_diagnosis, moderator_llm, mod_pipe):
    answer = query_model(moderator_llm, "\nHere is the correct diagnosis: " + correct_diagnosis + "\n Here was the doctor dialogue: " + diagnosis + "\nAre these the same?", "You are responsible for determining if the corrent diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else.")
    return answer.lower()


def extract_diagnosis_text(diagnosis_response):
    """Convert 'DIAGNOSIS READY: pneumonia' into 'pneumonia'.
    If the format is absent, return the full response rather than an empty string.
    """
    if diagnosis_response is None:
        return ""
    text = normalize_answer(diagnosis_response)
    match = re.search(r"DIAGNOSIS\s+READY\s*:\s*(.*)", text, flags=re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1))
    return text


def force_doctor_final_diagnosis(doctor_llm, scenario, full_dialogue):
    """Force a final diagnosis when the doctor loop ends without DIAGNOSIS READY."""
    prompt = (
        "\nHere is the complete dialogue so far:\n"
        + full_dialogue
        + "\n\nThe interviewing period is over. Give exactly one final diagnosis now."
    )
    system_prompt = (
        "You are the interviewing doctor. You must now provide your final diagnosis. "
        "Your answer must use exactly this format: \"DIAGNOSIS READY: [diagnosis here]\". "
        "Do not ask more questions."
        "\n\nInitial objective/context for the doctor:\n"
        + str(scenario.examiner_information())
    )
    return query_model(doctor_llm, prompt, system_prompt, clip_prompt=True)


def main(
    api_key,
    replicate_api_key,
    inf_type,
    doctor_bias,
    patient_bias,
    doctor_llm,
    patient_llm,
    measurement_llm,
    moderator_llm,
    num_scenarios,
    dataset,
    img_request,
    total_inferences,
    anthropic_api_key=None,
    deepseek_api_key=None,
    doctor_prompt_json="doctor_prompts.json",
    doctor_prompt_style="default",
    icd10cm_jsonl="icd10cm_2026.jsonl",
    validate_icd10cm=False,
):
    if api_key is not None:
        os.environ["OPENAI_API_KEY"] = api_key
    if deepseek_api_key is not None:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key

    # Load doctor prompt template from JSON (falls back to built-in if default file is absent)
    doctor_prompt_template = None
    try:
        doctor_prompt_template = load_doctor_prompt_template(doctor_prompt_json, doctor_prompt_style)
        print(f"Using doctor prompt style: '{doctor_prompt_style}' from {doctor_prompt_json}")
    except FileNotFoundError as e:
        if doctor_prompt_style == "default" and doctor_prompt_json == "doctor_prompts.json":
            print(f"Warning: {e}. Falling back to built-in default doctor prompt.")
        else:
            raise

    diagnosis_output_format = "text"
    if doctor_prompt_template is not None:
        diagnosis_output_format = doctor_prompt_template.get("diagnosis_output_format", "text")

    icd10cm_dict = None
    if validate_icd10cm:
        icd10cm_dict = load_icd10cm_dictionary(icd10cm_jsonl)
        print(f"Loaded {len(icd10cm_dict)} ICD-10-CM codes from {icd10cm_jsonl}")

    anthropic_llms = ["claude3.5sonnet"]
    replicate_llms = ["llama-3-70b-instruct", "llama-2-70b-chat", "mixtral-8x7b"]
    if patient_llm in replicate_llms or doctor_llm in replicate_llms or measurement_llm in replicate_llms or moderator_llm in replicate_llms:
        if replicate_api_key is not None:
            os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    if doctor_llm in anthropic_llms or patient_llm in anthropic_llms or measurement_llm in anthropic_llms or moderator_llm in anthropic_llms:
        if anthropic_api_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # Load MedQA, MIMICIV or NEJM agent case scenarios
    if dataset == "MedQA":
        scenario_loader = ScenarioLoaderMedQA()
    elif dataset == "MedQA_Ext":
        scenario_loader = ScenarioLoaderMedQAExtended()
    elif dataset == "NEJM":
        scenario_loader = ScenarioLoaderNEJM()
    elif dataset == "NEJM_Ext":
        scenario_loader = ScenarioLoaderNEJMExtended()
    elif dataset == "MIMICIV":
        scenario_loader = ScenarioLoaderMIMICIV()
    else:
        raise Exception("Dataset {} does not exist".format(str(dataset)))
    total_correct = 0
    total_presents = 0

    # Pipeline for huggingface models
    if "HF_" in moderator_llm:
        pipe = load_huggingface_model(moderator_llm.replace("HF_", ""))
    else:
        pipe = None
    if num_scenarios is None: num_scenarios = scenario_loader.num_scenarios
    for _scenario_id in range(0, min(num_scenarios, scenario_loader.num_scenarios)):
        total_presents += 1
        pi_dialogue = str()
        # Initialize scenarios (MedQA/NEJM)
        scenario =  scenario_loader.get_scenario(id=_scenario_id)
        # Initialize agents
        meas_agent = MeasurementAgent(
            scenario=scenario,
            backend_str=measurement_llm)
        patient_agent = PatientAgent(
            scenario=scenario, 
            bias_present=patient_bias,
            backend_str=patient_llm)
        doctor_agent = DoctorAgent(
            scenario=scenario,
            bias_present=doctor_bias,
            backend_str=doctor_llm,
            max_infs=total_inferences,
            img_request=img_request,
            doctor_prompt_template=doctor_prompt_template)

        doctor_dialogue = ""
        for _inf_id in range(total_inferences):
            # Check for medical image request
            if dataset in ["NEJM", "NEJM_Ext"] and img_request:
                imgs = "REQUEST IMAGES" in doctor_dialogue
            else:
                imgs = False
            # Check if final inference
            if _inf_id == total_inferences - 1:
                pi_dialogue += "This is the final question. Please provide a diagnosis.\n"
            # Obtain doctor dialogue (human or llm agent)
            if inf_type == "human_doctor":
                doctor_dialogue = input("\nQuestion for patient: ")
            else: 
                doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue, image_requested=imgs)
            print("Doctor [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), doctor_dialogue)
            # Doctor has arrived at a diagnosis, check correctness
            if "DIAGNOSIS READY" in doctor_dialogue:
                if diagnosis_output_format == "icd10cm":
                    parsed_icd = parse_icd10cm_from_diagnosis(doctor_dialogue)
                    if validate_icd10cm:
                        parsed_icd = validate_icd10cm_output(parsed_icd, icd10cm_dict)
                    if not parsed_icd.get("icd10cm_valid", False):
                        print(f"Warning: invalid or unrecognized ICD-10-CM output: {doctor_dialogue}")
                    else:
                        code = parsed_icd.get("icd10cm_dictionary_code") or parsed_icd.get("icd10cm_code", "")
                        label = parsed_icd.get("icd10cm_dictionary_label") or parsed_icd.get("icd10cm_label", "")
                        print(f"  ICD-10-CM: {code} — {label}")
                moderator_answer = compare_results(doctor_dialogue, scenario.diagnosis_information(), moderator_llm, pipe)
                correctness = moderator_answer.startswith("yes")
                if correctness: total_correct += 1
                print("\nCorrect answer:", scenario.diagnosis_information())
                print("Scene {}, The diagnosis was ".format(_scenario_id), "CORRECT" if correctness else "INCORRECT", int((total_correct/total_presents)*100))
                break
            # Obtain medical exam from measurement reader
            if "REQUEST TEST" in doctor_dialogue:
                pi_dialogue = meas_agent.inference_measurement(doctor_dialogue,)
                print("Measurement [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
                patient_agent.add_hist(pi_dialogue)
            # Obtain response from patient
            else:
                if inf_type == "human_patient":
                    pi_dialogue = input("\nResponse to doctor: ")
                else:
                    pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
                print("Patient [{}%]:".format(int(((_inf_id+1)/total_inferences)*100)), pi_dialogue)
                meas_agent.add_hist(pi_dialogue)
            # Prevent API timeouts
            time.sleep(1.0)

    print("\n==============================")
    print("Final Evaluation Results")
    print("==============================")
    print(f"Dataset: {dataset}")
    print(f"Doctor prompt style: {doctor_prompt_style}")
    print(f"Doctor prompt JSON: {doctor_prompt_json}")
    print(f"Diagnosis output format: {diagnosis_output_format}")
    if validate_icd10cm and icd10cm_dict is not None:
        print(f"ICD-10-CM dictionary: {icd10cm_jsonl} ({len(icd10cm_dict)} codes)")
    print(f"Total scenarios evaluated: {total_presents}")
    print(f"Correct diagnoses: {total_correct}")
    print(f"Incorrect diagnoses: {total_presents - total_correct}")

    if total_presents > 0:
        final_accuracy = total_correct / total_presents
        print(f"Final Accuracy: {final_accuracy:.4f}")
        print(f"Final Accuracy Percentage: {final_accuracy * 100:.2f}%")
    else:
        print("Final Accuracy: N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Diagnosis Simulation CLI")
    parser.add_argument("--openai_api_key", type=str, required=False, help="OpenAI API Key")
    parser.add_argument("--deepseek_api_key", type=str, required=False, help="DeepSeek API Key")
    parser.add_argument("--replicate_api_key", type=str, required=False, help="Replicate API Key")
    parser.add_argument("--inf_type", type=str, choices=["llm", "human_doctor", "human_patient"], default="llm")
    parser.add_argument("--doctor_bias", type=str, help="Doctor bias type", default="None", choices=["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument("--patient_bias", type=str, help="Patient bias type", default="None", choices=["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])

    # DeepSeek defaults. You can also use: deepseek-chat, deepseek-reasoner, deepseek-v4-pro.
    parser.add_argument("--doctor_llm", type=str, default="deepseek-v4-pro")
    parser.add_argument("--patient_llm", type=str, default="deepseek-v4-flash")
    parser.add_argument("--measurement_llm", type=str, default="deepseek-v4-flash")
    parser.add_argument("--moderator_llm", type=str, default="deepseek-v4-flash")

    parser.add_argument("--agent_dataset", type=str, default="MedQA")  # MedQA, MedQA_Ext, NEJM, NEJM_Ext, MIMICIV
    parser.add_argument("--doctor_image_request", action="store_true", help="Enable image request path. Do not use this with DeepSeek.")
    parser.add_argument("--num_scenarios", type=int, default=None, required=False, help="Number of scenarios to simulate")
    parser.add_argument("--total_inferences", type=int, default=20, required=False, help="Number of inferences between patient and doctor")
    parser.add_argument("--anthropic_api_key", type=str, default=None, required=False, help="Anthropic API key for Claude 3.5 Sonnet")
    parser.add_argument("--doctor_prompt_json", type=str, default="doctor_prompts.json", help="Path to JSON file containing doctor prompt templates")
    parser.add_argument("--doctor_prompt_style", type=str, default="default", help="Key/name of the doctor prompt style inside the JSON file")
    parser.add_argument("--icd10cm_jsonl", type=str, default="icd10cm_2026.jsonl", help="Path to ICD-10-CM JSONL code dictionary. Relative paths are resolved relative to agentclinic.py.")
    parser.add_argument("--validate_icd10cm", action="store_true", help="Validate ICD-10-CM diagnosis outputs against the ICD-10-CM JSONL dictionary.")

    args = parser.parse_args()

    main(
        args.openai_api_key,
        args.replicate_api_key,
        args.inf_type,
        args.doctor_bias,
        args.patient_bias,
        args.doctor_llm,
        args.patient_llm,
        args.measurement_llm,
        args.moderator_llm,
        args.num_scenarios,
        args.agent_dataset,
        args.doctor_image_request,
        args.total_inferences,
        args.anthropic_api_key,
        args.deepseek_api_key,
        args.doctor_prompt_json,
        args.doctor_prompt_style,
        args.icd10cm_jsonl,
        args.validate_icd10cm,
    )
