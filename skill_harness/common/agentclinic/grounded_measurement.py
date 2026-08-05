import json
import re
from dataclasses import dataclass


def normalize_test_name(value: str) -> str:
    text = str(value).lower()
    replacements = {
        r"\bcbc\b": "complete blood count",
        r"\blfts?\b": "liver function panel",
        r"\bcxr\b": "chest x ray",
        r"\bua\b": "urinalysis",
        r"\bekg\b": "electrocardiogram",
        r"\becg\b": "electrocardiogram",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    text = text.replace("x-ray", "x ray")
    text = re.sub(r"\bpelvis\b", "pelvic", text)
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def flatten_test_results(payload, path=()):
    if isinstance(payload, dict):
        rows = []
        for key, value in payload.items():
            rows.extend(flatten_test_results(value, path + (str(key),)))
        return rows
    return [(path, payload)]


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?:^| )({re.escape(phrase)})(?: |$)", text))


SPECIAL_GROUPS = {
    "liver function panel": {
        "bilirubin",
        "ast",
        "alt",
        "alkaline phosphatase",
        "albumin",
        "total protein",
        "tp activity",
    },
    "liver function tests": {
        "bilirubin",
        "ast",
        "alt",
        "alkaline phosphatase",
        "albumin",
        "total protein",
        "tp activity",
    },
    "coagulation studies": {
        "coagulation test results",
        "pt",
        "inr",
        "aptt",
        "act",
        "thrombin time",
        "fibrinogen",
    },
    "coagulation study": {
        "coagulation test results",
        "pt",
        "inr",
        "aptt",
        "act",
        "thrombin time",
        "fibrinogen",
    },
    "chest x ray": {"chest x ray", "chest radiograph"},
    "x ray": {"x ray", "radiograph"},
    "pelvic ultrasound": {"pelvic ultrasound"},
    "transvaginal ultrasound": {"pelvic ultrasound"},
    "transvaginal pelvic ultrasound": {"pelvic ultrasound"},
}


@dataclass(frozen=True)
class GroundedMeasurementEvent:
    request: str
    matched_paths: tuple[str, ...]
    unavailable: bool
    response: str

    def to_dict(self) -> dict:
        return {
            "request": self.request,
            "matched_paths": list(self.matched_paths),
            "unavailable": self.unavailable,
            "response": self.response,
        }


class GroundedMeasurementAgent:
    """Return only measurements that exist in the scenario's source data."""

    def __init__(self, scenario, missing_policy: str = "unavailable"):
        if missing_policy != "unavailable":
            raise ValueError("Grounded measurement only supports missing_policy='unavailable'")
        self.scenario = scenario
        self.missing_policy = missing_policy
        self.agent_hist = ""
        self.source_information = scenario.exam_information()
        self.events: list[GroundedMeasurementEvent] = []

    def add_hist(self, hist_str) -> None:
        self.agent_hist += str(hist_str) + "\n\n"

    def inference_measurement(self, question) -> str:
        request = str(question).split("REQUEST TEST:", 1)[-1].strip()
        normalized_request = normalize_test_name(request)
        matched = {}

        for path, value in flatten_test_results(self.source_information):
            display_path = path[1:] if path and path[0] == "tests" else path
            normalized_parts = [normalize_test_name(part) for part in display_path]
            normalized_path = " ".join(part for part in normalized_parts if part)
            leaf = normalized_parts[-1] if normalized_parts else ""

            direct_match = bool(leaf) and _contains_phrase(normalized_request, leaf)
            group_match = any(
                _contains_phrase(normalized_request, trigger)
                and any(_contains_phrase(normalized_path, pattern) for pattern in patterns)
                for trigger, patterns in SPECIAL_GROUPS.items()
            )
            top_level_match = bool(normalized_parts) and _contains_phrase(
                normalized_request, normalized_parts[0]
            )
            if direct_match or group_match or top_level_match:
                matched[".".join(display_path)] = value

        if matched:
            serialized = json.dumps(matched, ensure_ascii=False, sort_keys=True)
            response = (
                f"RESULTS: {serialized}. "
                "Only these source-backed measurements are available; any other "
                "requested measurement is unavailable and must not be treated as normal."
            )
        else:
            response = (
                "RESULTS UNAVAILABLE: No requested measurement is present in the "
                "case data. Do not infer a normal result."
            )

        event = GroundedMeasurementEvent(
            request=request,
            matched_paths=tuple(sorted(matched)),
            unavailable=not bool(matched),
            response=response,
        )
        self.events.append(event)
        self.agent_hist += str(question) + "\n\n" + response + "\n\n"
        return response

    def metadata(self) -> dict:
        return {
            "mode": "grounded_lookup",
            "request_count": len(self.events),
            "unavailable_count": sum(event.unavailable for event in self.events),
            "events": [event.to_dict() for event in self.events],
        }
