import argparse
import os
import re
import random
import time
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed; .env will not be auto-loaded. "
          "Run `pip install python-dotenv` or set keys via export/CLI flags.")

# Datasets live in the repo-root AgentClinic/ package, one level up from this module.
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AgentClinic"
)

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
}

DEEPSEEK_MODELS = {
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
}


def load_huggingface_model(model_name):
    if pipeline is None:
        raise ImportError("transformers is not installed. Run: pip install transformers")
    pipe = pipeline("text-generation", model=model_name, device_map="auto")
    return pipe


def inference_huggingface(prompt, pipe):
    response = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
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
                    temperature=0,
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

                response = client.chat.completions.create(
                    model=api_model,
                    messages=messages,
                    temperature=0,
                    max_tokens=200,
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
                    max_tokens=256,
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
                    input={"prompt": prompt, "system_prompt": system_prompt, "max_new_tokens": 200},
                )
                return normalize_answer("".join(output))

            if model_str == "mixtral-8x7b":
                if replicate is None:
                    raise ImportError("replicate is not installed. Run: pip install replicate")
                output = replicate.run(
                    mixtral_url,
                    input={"prompt": prompt, "system_prompt": system_prompt, "max_new_tokens": 75},
                )
                return normalize_answer("".join(output))

            if model_str == "llama-3-70b-instruct":
                if replicate is None:
                    raise ImportError("replicate is not installed. Run: pip install replicate")
                output = replicate.run(
                    llama3_url,
                    input={"prompt": prompt, "system_prompt": system_prompt, "max_new_tokens": 200},
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
        with open(os.path.join(DATA_DIR, "agentclinic_medqa.jsonl"), "r") as f:
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
        with open(os.path.join(DATA_DIR, "agentclinic_medqa_extended.jsonl"), "r") as f:
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
        with open(os.path.join(DATA_DIR, "agentclinic_mimiciv.jsonl"), "r") as f:
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
        with open(os.path.join(DATA_DIR, "agentclinic_nejm_extended.jsonl"), "r") as f:
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
        with open(os.path.join(DATA_DIR, "agentclinic_nejm.jsonl"), "r") as f:
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
    def __init__(self, scenario, backend_str="gpt4", max_infs=20, bias_present=None, img_request=False, output_format="normal") -> None:
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
        self.output_format = output_format
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
        base = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {} questions total before you must make a decision. You have asked {} questions so far. You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\"".format(self.MAX_INFS, self.infs) + ("You may also request medical images related to the disease to be returned with \"REQUEST IMAGES\"." if self.img_request else "")
        if self.output_format == "anchor_compare":
            base += "\nFor version-equivalence testing, when you are ready to diagnose, use exactly this structured format:\nDIAGNOSIS READY: <single most likely diagnosis>\nCANDIDATES: <diagnosis 1>; <diagnosis 2>; <diagnosis 3>\nKEY EVIDENCE: <brief evidence separated by semicolons>"
        presentation = "\n\nBelow is all of the information you have. {}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams.".format(self.presentation)
        return base + bias_prompt + presentation

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



# ============================================================
# Anchor-case behavioral equivalence utilities
# ============================================================

def normalize_diagnosis_name(text):
    """Normalize diagnosis strings for consistency comparison.

    This is intentionally not an accuracy check. It only canonicalizes surface form
    differences so two versions can be compared more fairly.
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"diagnosis ready\s*:", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"final diagnosis\s*:", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9\s\-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_final_diagnosis(dialogue_text):
    """Extract the final diagnosis from a doctor response or dialogue transcript."""
    if dialogue_text is None:
        return ""
    text = str(dialogue_text)

    # Preferred structured format:
    # DIAGNOSIS READY: pneumonia
    match = re.search(
        r"DIAGNOSIS READY\s*:\s*(.*?)(?:\n|CANDIDATES\s*:|KEY EVIDENCE\s*:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return normalize_diagnosis_name(match.group(1))

    # Fallback: take the last non-empty line.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return normalize_diagnosis_name(lines[-1])


def extract_candidate_diagnoses(dialogue_text):
    """Extract candidate diagnoses from the structured CANDIDATES line."""
    if dialogue_text is None:
        return []
    text = str(dialogue_text)
    match = re.search(
        r"CANDIDATES\s*:\s*(.*?)(?:\n|KEY EVIDENCE\s*:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        final_dx = extract_final_diagnosis(text)
        return [final_dx] if final_dx else []

    raw = match.group(1)
    parts = re.split(r";|,|\n|\d+[\).\s]+", raw)
    candidates = []
    for part in parts:
        dx = normalize_diagnosis_name(part)
        if dx and dx not in candidates:
            candidates.append(dx)
    return candidates


def extract_key_evidence(dialogue_text):
    """Extract key evidence items from the structured KEY EVIDENCE line."""
    if dialogue_text is None:
        return []
    text = str(dialogue_text)
    match = re.search(r"KEY EVIDENCE\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    raw = match.group(1)
    parts = re.split(r";|\n|\d+[\).\s]+", raw)
    evidence = []
    for part in parts:
        item = normalize_answer(part).lower()
        if item and item not in evidence:
            evidence.append(item)
    return evidence


def jaccard_similarity(items_a, items_b):
    set_a = set([x for x in items_a if x])
    set_b = set([x for x in items_b if x])
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))


def js_divergence_from_counts(counts_a, counts_b):
    """Jensen-Shannon divergence using diagnosis-count dictionaries.

    Returns a value in [0, 1] when log base 2 is used.
    0 means identical empirical output distributions.
    """
    import math

    keys = sorted(set(counts_a.keys()).union(set(counts_b.keys())))
    if not keys:
        return 0.0

    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    if total_a == 0 and total_b == 0:
        return 0.0
    if total_a == 0 or total_b == 0:
        return 1.0

    p = [counts_a.get(k, 0) / total_a for k in keys]
    q = [counts_b.get(k, 0) / total_b for k in keys]
    m = [(x + y) / 2 for x, y in zip(p, q)]

    def kl_divergence(x, y):
        value = 0.0
        for xi, yi in zip(x, y):
            if xi > 0:
                value += xi * math.log(xi / yi, 2)
        return value

    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def update_count(counts, key):
    key = key if key else "<missing diagnosis>"
    counts[key] = counts.get(key, 0) + 1


def most_common_key(counts):
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]


def compare_diagnosis_equivalence(dx_a, dx_b, moderator_llm, use_moderator=True):
    """Compare two model outputs with no reference to the ground-truth answer."""
    dx_a = normalize_diagnosis_name(dx_a)
    dx_b = normalize_diagnosis_name(dx_b)
    if not dx_a or not dx_b:
        return False
    if dx_a == dx_b:
        return True
    if not use_moderator:
        return False

    prompt = (
        "\nDiagnosis from version A: " + dx_a +
        "\nDiagnosis from version B: " + dx_b +
        "\nAre these the same disease or clinically equivalent diagnoses?"
    )
    answer = query_model(
        moderator_llm,
        prompt,
        "You determine whether two diagnosis labels refer to the same disease. "
        "Do not use any ground-truth answer. Respond only with Yes or No.",
    )
    return normalize_answer(answer).lower().startswith("yes")


def run_agentclinic_case(
    scenario,
    dataset,
    inf_type,
    doctor_bias,
    patient_bias,
    doctor_llm,
    patient_llm,
    measurement_llm,
    total_inferences,
    img_request,
    output_format="anchor_compare",
    verbose=False,
):
    """Run one scenario once and return structured behavioral outputs.

    This does not check correctness against the scenario's correct diagnosis.
    It only records the agent behavior for later version-to-version comparison.
    """
    pi_dialogue = str()
    meas_agent = MeasurementAgent(scenario=scenario, backend_str=measurement_llm)
    patient_agent = PatientAgent(
        scenario=scenario,
        bias_present=patient_bias,
        backend_str=patient_llm,
    )
    doctor_agent = DoctorAgent(
        scenario=scenario,
        bias_present=doctor_bias,
        backend_str=doctor_llm,
        max_infs=total_inferences,
        img_request=img_request,
        output_format=output_format,
    )

    doctor_dialogue = ""
    transcript = []

    for _inf_id in range(total_inferences):
        if dataset in ["NEJM", "NEJM_Ext"] and img_request:
            imgs = "REQUEST IMAGES" in doctor_dialogue
        else:
            imgs = False

        if _inf_id == total_inferences - 1:
            pi_dialogue += "This is the final question. Please provide a diagnosis in the required structured format.\n"

        if inf_type == "human_doctor":
            doctor_dialogue = input("\nQuestion for patient: ")
        else:
            doctor_dialogue = doctor_agent.inference_doctor(pi_dialogue, image_requested=imgs)

        transcript.append({"role": "doctor", "content": doctor_dialogue})
        if verbose:
            print("Doctor [{}%]:".format(int(((_inf_id + 1) / total_inferences) * 100)), doctor_dialogue)

        if "DIAGNOSIS READY" in doctor_dialogue:
            break

        if "REQUEST TEST" in doctor_dialogue:
            pi_dialogue = meas_agent.inference_measurement(doctor_dialogue)
            transcript.append({"role": "measurement", "content": pi_dialogue})
            patient_agent.add_hist(pi_dialogue)
            if verbose:
                print("Measurement [{}%]:".format(int(((_inf_id + 1) / total_inferences) * 100)), pi_dialogue)
        else:
            if inf_type == "human_patient":
                pi_dialogue = input("\nResponse to doctor: ")
            else:
                pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
            transcript.append({"role": "patient", "content": pi_dialogue})
            meas_agent.add_hist(pi_dialogue)
            if verbose:
                print("Patient [{}%]:".format(int(((_inf_id + 1) / total_inferences) * 100)), pi_dialogue)

        time.sleep(1.0)

    final_dx = extract_final_diagnosis(doctor_dialogue)
    candidates = extract_candidate_diagnoses(doctor_dialogue)
    evidence = extract_key_evidence(doctor_dialogue)

    # Diagnostic: when extraction failed, dump enough info to distinguish
    # (a) empty API response, (b) truncation before DIAGNOSIS READY,
    # (c) truncation right at DIAGNOSIS READY:, (d) parser miss on present diagnosis.
    if not final_dx:
        msg = doctor_dialogue or ""
        marker_present = "DIAGNOSIS READY" in msg.upper()
        tail = msg[-200:].replace("\n", "\\n")
        print(
            "  [empty final_dx] msg_len={} chars | has 'DIAGNOSIS READY'={} | last_200={!r}".format(
                len(msg), marker_present, tail
            )
        )

    return {
        "final_doctor_message": doctor_dialogue,
        "final_diagnosis": final_dx,
        "candidate_diagnoses": candidates,
        "key_evidence": evidence,
        "transcript": transcript,
    }


def load_scenario_loader(dataset):
    if dataset == "MedQA":
        return ScenarioLoaderMedQA()
    if dataset == "MedQA_Ext":
        return ScenarioLoaderMedQAExtended()
    if dataset == "NEJM":
        return ScenarioLoaderNEJM()
    if dataset == "NEJM_Ext":
        return ScenarioLoaderNEJMExtended()
    if dataset == "MIMICIV":
        return ScenarioLoaderMIMICIV()
    raise Exception("Dataset {} does not exist".format(str(dataset)))


def run_anchor_comparison(
    api_key,
    replicate_api_key,
    inf_type,
    doctor_bias,
    patient_bias,
    baseline_doctor_llm,
    candidate_doctor_llm,
    patient_llm,
    measurement_llm,
    moderator_llm,
    num_scenarios,
    dataset,
    img_request,
    total_inferences,
    anthropic_api_key=None,
    deepseek_api_key=None,
    runs_per_case=3,
    agreement_threshold=0.90,
    jaccard_threshold=0.75,
    jsd_threshold=0.10,
    output_json=None,
    use_moderator_for_equivalence=True,
    verbose=False,
):
    """Compare two doctor-agent versions on fixed anchor cases.

    This mode does NOT compute diagnostic accuracy. It treats the selected
    scenarios as anchor cases and measures behavioral equivalence between the
    baseline doctor model and candidate doctor model.
    """
    if api_key is not None:
        os.environ["OPENAI_API_KEY"] = api_key
    if deepseek_api_key is not None:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key

    anthropic_llms = ["claude3.5sonnet"]
    replicate_llms = ["llama-3-70b-instruct", "llama-2-70b-chat", "mixtral-8x7b"]
    all_llms = [
        baseline_doctor_llm,
        candidate_doctor_llm,
        patient_llm,
        measurement_llm,
        moderator_llm,
    ]
    if any(llm in replicate_llms for llm in all_llms) and replicate_api_key is not None:
        os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
    if any(llm in anthropic_llms for llm in all_llms) and anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    scenario_loader = load_scenario_loader(dataset)
    if num_scenarios is None:
        num_scenarios = scenario_loader.num_scenarios
    num_scenarios = min(num_scenarios, scenario_loader.num_scenarios)

    case_results = []
    top1_same_count = 0
    candidate_jaccards = []
    evidence_jaccards = []
    jsds = []

    print("\n==============================")
    print("Anchor-Case Version Comparison")
    print("==============================")
    print(f"Dataset: {dataset}")
    print(f"Anchor cases: {num_scenarios}")
    print(f"Runs per case per version: {runs_per_case}")
    print(f"Baseline doctor: {baseline_doctor_llm}")
    print(f"Candidate doctor: {candidate_doctor_llm}")
    print("Metric focus: behavior consistency, not accuracy")

    for scenario_id in range(num_scenarios):
        scenario = scenario_loader.get_scenario(id=scenario_id)

        baseline_runs = []
        candidate_runs = []
        baseline_counts = {}
        candidate_counts = {}

        print(f"\n--- Anchor case {scenario_id} ---")

        for run_id in range(runs_per_case):
            baseline_result = run_agentclinic_case(
                scenario=scenario,
                dataset=dataset,
                inf_type=inf_type,
                doctor_bias=doctor_bias,
                patient_bias=patient_bias,
                doctor_llm=baseline_doctor_llm,
                patient_llm=patient_llm,
                measurement_llm=measurement_llm,
                total_inferences=total_inferences,
                img_request=img_request,
                output_format="anchor_compare",
                verbose=verbose,
            )
            baseline_runs.append(baseline_result)
            update_count(baseline_counts, baseline_result["final_diagnosis"])
            print(f"Baseline run {run_id + 1}: {baseline_result['final_diagnosis']}")

            candidate_result = run_agentclinic_case(
                scenario=scenario,
                dataset=dataset,
                inf_type=inf_type,
                doctor_bias=doctor_bias,
                patient_bias=patient_bias,
                doctor_llm=candidate_doctor_llm,
                patient_llm=patient_llm,
                measurement_llm=measurement_llm,
                total_inferences=total_inferences,
                img_request=img_request,
                output_format="anchor_compare",
                verbose=verbose,
            )
            candidate_runs.append(candidate_result)
            update_count(candidate_counts, candidate_result["final_diagnosis"])
            print(f"Candidate run {run_id + 1}: {candidate_result['final_diagnosis']}")

        baseline_majority = most_common_key(baseline_counts)
        candidate_majority = most_common_key(candidate_counts)
        top1_same = compare_diagnosis_equivalence(
            baseline_majority,
            candidate_majority,
            moderator_llm,
            use_moderator=use_moderator_for_equivalence,
        )

        baseline_candidates = []
        candidate_candidates = []
        baseline_evidence = []
        candidate_evidence = []

        for item in baseline_runs:
            baseline_candidates.extend(item["candidate_diagnoses"])
            baseline_evidence.extend(item["key_evidence"])
        for item in candidate_runs:
            candidate_candidates.extend(item["candidate_diagnoses"])
            candidate_evidence.extend(item["key_evidence"])

        cand_jaccard = jaccard_similarity(baseline_candidates, candidate_candidates)
        evid_jaccard = jaccard_similarity(baseline_evidence, candidate_evidence)
        jsd = js_divergence_from_counts(baseline_counts, candidate_counts)

        top1_same_count += int(top1_same)
        candidate_jaccards.append(cand_jaccard)
        evidence_jaccards.append(evid_jaccard)
        jsds.append(jsd)

        case_summary = {
            "scenario_id": scenario_id,
            "baseline_counts": baseline_counts,
            "candidate_counts": candidate_counts,
            "baseline_majority_diagnosis": baseline_majority,
            "candidate_majority_diagnosis": candidate_majority,
            "top1_equivalent": top1_same,
            "candidate_jaccard": cand_jaccard,
            "evidence_jaccard": evid_jaccard,
            "js_divergence": jsd,
            "baseline_runs": baseline_runs,
            "candidate_runs": candidate_runs,
        }
        case_results.append(case_summary)

        print(f"Majority baseline:  {baseline_majority}")
        print(f"Majority candidate: {candidate_majority}")
        print(f"Top-1 equivalent:  {top1_same}")
        print(f"Candidate Jaccard: {cand_jaccard:.4f}")
        print(f"Evidence Jaccard:  {evid_jaccard:.4f}")
        print(f"JS Divergence:     {jsd:.4f}")

    top1_agreement = top1_same_count / num_scenarios if num_scenarios else 0.0
    avg_candidate_jaccard = sum(candidate_jaccards) / len(candidate_jaccards) if candidate_jaccards else 0.0
    avg_evidence_jaccard = sum(evidence_jaccards) / len(evidence_jaccards) if evidence_jaccards else 0.0
    avg_jsd = sum(jsds) / len(jsds) if jsds else 1.0

    behaviorally_equivalent = (
        top1_agreement >= agreement_threshold
        and avg_candidate_jaccard >= jaccard_threshold
        and avg_jsd <= jsd_threshold
    )

    summary = {
        "dataset": dataset,
        "num_anchor_cases": num_scenarios,
        "runs_per_case": runs_per_case,
        "baseline_doctor_llm": baseline_doctor_llm,
        "candidate_doctor_llm": candidate_doctor_llm,
        "top1_agreement": top1_agreement,
        "average_candidate_jaccard": avg_candidate_jaccard,
        "average_evidence_jaccard": avg_evidence_jaccard,
        "average_js_divergence": avg_jsd,
        "thresholds": {
            "top1_agreement_min": agreement_threshold,
            "candidate_jaccard_min": jaccard_threshold,
            "js_divergence_max": jsd_threshold,
        },
        "behaviorally_equivalent": behaviorally_equivalent,
        "case_results": case_results,
    }

    print("\n==============================")
    print("Final Behavioral Equivalence Results")
    print("==============================")
    print(f"Top-1 diagnosis agreement:       {top1_agreement:.4f}")
    print(f"Avg candidate Jaccard:           {avg_candidate_jaccard:.4f}")
    print(f"Avg evidence Jaccard:            {avg_evidence_jaccard:.4f}")
    print(f"Avg JS divergence:               {avg_jsd:.4f}")
    print(f"Behaviorally equivalent?:        {behaviorally_equivalent}")
    print("Note: This is not accuracy. No ground-truth diagnosis is used for the final decision.")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved comparison report to: {output_json}")

    return summary


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
):
    if api_key is not None:
        os.environ["OPENAI_API_KEY"] = api_key
    if deepseek_api_key is not None:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key

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
            img_request=img_request)

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

    # DeepSeek defaults: deepseek-v4-flash (cheap, non-thinking) and deepseek-v4-pro (premium).
    # Legacy aliases deepseek-chat / deepseek-reasoner still work but map to v4-flash; prefer the explicit v4 names.
    parser.add_argument("--doctor_llm", type=str, default="deepseek-v4-pro")
    parser.add_argument("--patient_llm", type=str, default="deepseek-v4-flash")
    parser.add_argument("--measurement_llm", type=str, default="deepseek-v4-flash")
    parser.add_argument("--moderator_llm", type=str, default="deepseek-v4-flash")

    parser.add_argument("--agent_dataset", type=str, default="MedQA")  # MedQA, MedQA_Ext, NEJM, NEJM_Ext, MIMICIV
    parser.add_argument("--doctor_image_request", action="store_true", help="Enable image request path. Do not use this with DeepSeek.")
    parser.add_argument("--num_scenarios", type=int, default=None, required=False, help="Number of scenarios to simulate")
    parser.add_argument("--total_inferences", type=int, default=20, required=False, help="Number of inferences between patient and doctor")
    parser.add_argument("--anthropic_api_key", type=str, default=None, required=False, help="Anthropic API key for Claude 3.5 Sonnet")

    # Version-equivalence / anchor-case comparison mode.
    # This mode compares two doctor-agent versions and does NOT use diagnostic accuracy.
    parser.add_argument("--eval_mode", type=str, default="accuracy", choices=["accuracy", "anchor_compare"])
    parser.add_argument("--baseline_doctor_llm", type=str, default=None, help="Baseline doctor model for anchor_compare mode")
    parser.add_argument("--candidate_doctor_llm", type=str, default=None, help="Candidate doctor model for anchor_compare mode")
    parser.add_argument("--runs_per_case", type=int, default=3, help="Repeated runs per anchor case per version")
    parser.add_argument("--agreement_threshold", type=float, default=0.90, help="Minimum Top-1 diagnosis agreement for equivalence")
    parser.add_argument("--candidate_jaccard_threshold", type=float, default=0.75, help="Minimum candidate-diagnosis Jaccard similarity for equivalence")
    parser.add_argument("--jsd_threshold", type=float, default=0.10, help="Maximum Jensen-Shannon divergence for equivalence")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save anchor comparison report as JSON")
    parser.add_argument("--no_moderator_equivalence", action="store_true", help="Use exact normalized diagnosis match instead of moderator equivalence")
    parser.add_argument("--verbose_compare", action="store_true", help="Print full turn-by-turn dialogue in anchor_compare mode")

    args = parser.parse_args()

    if args.eval_mode == "anchor_compare":
        baseline_doctor_llm = args.baseline_doctor_llm or args.doctor_llm
        candidate_doctor_llm = args.candidate_doctor_llm or args.doctor_llm

        run_anchor_comparison(
            args.openai_api_key,
            args.replicate_api_key,
            args.inf_type,
            args.doctor_bias,
            args.patient_bias,
            baseline_doctor_llm,
            candidate_doctor_llm,
            args.patient_llm,
            args.measurement_llm,
            args.moderator_llm,
            args.num_scenarios,
            args.agent_dataset,
            args.doctor_image_request,
            args.total_inferences,
            args.anthropic_api_key,
            args.deepseek_api_key,
            args.runs_per_case,
            args.agreement_threshold,
            args.candidate_jaccard_threshold,
            args.jsd_threshold,
            args.output_json,
            not args.no_moderator_equivalence,
            args.verbose_compare,
        )
    else:
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
        )
