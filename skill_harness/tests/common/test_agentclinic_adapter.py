import tempfile
import unittest
from pathlib import Path

from AgentClinic.agentclinic import DoctorAgent, MeasurementAgent
from skill_harness.artifacts import HarnessArtifact, SkillArtifact
from skill_harness.common.agentclinic import AgentClinicAdapter
from skill_harness.common.agentclinic.skill_doctor import SkillDoctorAgent


class DummyScenario:
    def examiner_information(self):
        return "A patient with fatigue."

    def exam_information(self):
        return {
            "Vital_Signs": {
                "Blood_Pressure": "137/98 mmHg",
                "Heart_Rate": "120/min",
            },
            "tests": {
                "Coagulation_Test_Results": {
                    "ACT": "52.0 s (prolonged)",
                    "PT": "14.0 s",
                },
                "Blood_Work": {
                    "Bilirubin": "25 mg/dL (elevated)",
                    "AST": "600 IU/L (elevated)",
                    "ALT": "650 IU/L (elevated)",
                    "INR": "1.5 (elevated)",
                },
                "Imaging": {
                    "Pelvic_Ultrasound": {
                        "Findings": "Left adnexal mass",
                    }
                },
            }
        }


class AgentClinicSkillAdapterTest(unittest.TestCase):
    def setUp(self):
        self.scenario = DummyScenario()
        self.config = {"doctor_llm": "unused", "total_inferences": 3}

    def make_skill(self):
        return SkillArtifact(
            skill_id="diagnostic_reasoning",
            version="test_seed",
            content="Prefer discriminative questions before broad testing.",
            sha256="test-sha",
            path=Path("test_seed.md"),
            generated_by="test",
        )

    def test_baseline_uses_unmodified_doctor_agent(self):
        doctor = AgentClinicAdapter().build_doctor(self.scenario, self.config)
        self.assertIs(type(doctor), DoctorAgent)
        self.assertNotIn("<DOMAIN_SKILL", doctor.system_prompt())

    def test_skill_adapter_uses_external_subclass_and_preserves_protocol(self):
        doctor = AgentClinicAdapter(self.make_skill()).build_doctor(
            self.scenario, self.config
        )
        prompt = doctor.system_prompt()

        self.assertIs(type(doctor), SkillDoctorAgent)
        skill_end = prompt.index("</DOMAIN_SKILL>")
        protocol_start = prompt.index("<IMMUTABLE_PROTOCOL>")
        self.assertLess(skill_end, protocol_start)
        self.assertIn('"REQUEST TEST: [test]"', prompt[protocol_start:])
        self.assertIn('"DIAGNOSIS READY: [diagnosis here]"', prompt[protocol_start:])

    def test_skill_artifact_loads_version_metadata_and_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_path = Path(temp_dir) / "v003.md"
            metadata_path = Path(temp_dir) / "v003.meta"
            skill_path.write_text("A compact skill.", encoding="utf-8")
            metadata_path.write_text(
                '{"skill_id":"clinical","version":"v003","generated_by":"skillopt-lite"}',
                encoding="utf-8",
            )

            artifact = SkillArtifact.load(skill_path)

        self.assertEqual(artifact.skill_id, "clinical")
        self.assertEqual(artifact.version, "v003")
        self.assertEqual(artifact.generated_by, "skillopt-lite")
        self.assertEqual(len(artifact.sha256), 64)

    def test_harness_changes_orchestration_without_changing_core(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            harness_path = Path(temp_dir) / "test.toml"
            harness_path.write_text(
                '[harness]\nid="test"\nversion="test"\n'
                'agent_system="agentclinic"\n[doctor]\nmax_inferences=8\n',
                encoding="utf-8",
            )
            harness = HarnessArtifact.load(harness_path)
            adapter = AgentClinicAdapter(harness_artifact=harness)
            doctor = adapter.build_doctor(self.scenario, self.config)
            effective = adapter.effective_config(self.config)

        self.assertIs(type(doctor), DoctorAgent)
        self.assertEqual(effective["total_inferences"], 8)
        self.assertEqual(doctor.MAX_INFS, 8)
        self.assertEqual(adapter.variant_metadata()["harness"]["version"], "test")

    def test_skill_and_harness_compose_as_independent_dimensions(self):
        harness = HarnessArtifact(
            harness_id="test_harness",
            version="v001",
            agent_system="agentclinic",
            config={"doctor": {"max_inferences": 5}},
            sha256="harness-sha",
            path=Path("v001.toml"),
            generated_by="test",
        )
        adapter = AgentClinicAdapter(self.make_skill(), harness)
        doctor = adapter.build_doctor(self.scenario, self.config)
        metadata = adapter.variant_metadata()

        self.assertIs(type(doctor), SkillDoctorAgent)
        self.assertEqual(doctor.MAX_INFS, 5)
        self.assertEqual(metadata["skill"]["version"], "test_seed")
        self.assertEqual(metadata["harness"]["version"], "v001")

    def test_pure_baseline_keeps_generative_measurement(self):
        adapter = AgentClinicAdapter()
        config = {**self.config, "measurement_llm": "unused"}
        self.assertIs(
            type(adapter.build_measurement(self.scenario, config)), MeasurementAgent
        )


if __name__ == "__main__":
    unittest.main()
