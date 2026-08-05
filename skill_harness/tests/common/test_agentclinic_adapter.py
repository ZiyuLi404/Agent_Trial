import tempfile
import unittest
from pathlib import Path

from AgentClinic.agentclinic import DoctorAgent, MeasurementAgent
from skill_harness.artifacts import HarnessArtifact, SkillArtifact
from skill_harness.common.agentclinic import AgentClinicAdapter
from skill_harness.common.agentclinic.grounded_measurement import GroundedMeasurementAgent
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
            version="v000",
            content="Prefer discriminative questions before broad testing.",
            sha256="test-sha",
            path=Path("v000.md"),
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
        harness_path = (
            Path(__file__).resolve().parents[2]
            / "artifacts"
            / "harnesses"
            / "baseline"
            / "diagnostic_efficiency_v000.toml"
        )
        harness = HarnessArtifact.load(harness_path)
        adapter = AgentClinicAdapter(harness_artifact=harness)
        doctor = adapter.build_doctor(self.scenario, self.config)
        effective = adapter.effective_config(self.config)

        self.assertIs(type(doctor), DoctorAgent)
        self.assertEqual(effective["total_inferences"], 8)
        self.assertEqual(doctor.MAX_INFS, 8)
        self.assertIn("Prioritize reaching the most likely diagnosis quickly", doctor.system_prompt())
        self.assertEqual(adapter.variant_metadata()["harness"]["version"], "v000")

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
        self.assertEqual(metadata["skill"]["version"], "v000")
        self.assertEqual(metadata["harness"]["version"], "v001")

    def test_grounded_measurement_never_converts_missing_factor_to_normal(self):
        measurement = GroundedMeasurementAgent(self.scenario)

        response = measurement.inference_measurement(
            "REQUEST TEST: aPTT and Factor VIII activity level"
        )

        self.assertIn("RESULTS UNAVAILABLE", response)
        self.assertIn("Do not infer a normal result", response)
        self.assertNotIn("NORMAL READINGS", response)
        self.assertEqual(measurement.metadata()["unavailable_count"], 1)

    def test_grounded_measurement_returns_only_source_backed_values(self):
        measurement = GroundedMeasurementAgent(self.scenario)

        coagulation = measurement.inference_measurement(
            "REQUEST TEST: coagulation studies including ACT and PT"
        )
        liver = measurement.inference_measurement(
            "REQUEST TEST: liver function panel and INR"
        )
        vital_signs = measurement.inference_measurement("REQUEST TEST: vital signs")
        pelvic = measurement.inference_measurement(
            "REQUEST TEST: transvaginal pelvis ultrasound"
        )

        self.assertIn('"Coagulation_Test_Results.ACT": "52.0 s (prolonged)"', coagulation)
        self.assertIn('"Coagulation_Test_Results.PT": "14.0 s"', coagulation)
        self.assertNotIn("Factor VIII", coagulation)
        self.assertIn('"Blood_Work.ALT": "650 IU/L (elevated)"', liver)
        self.assertIn('"Blood_Work.INR": "1.5 (elevated)"', liver)
        self.assertIn('"Vital_Signs.Heart_Rate": "120/min"', vital_signs)
        self.assertIn(
            '"Imaging.Pelvic_Ultrasound.Findings": "Left adnexal mass"',
            pelvic,
        )

    def test_measurement_mode_is_versioned_by_harness(self):
        artifact_root = (
            Path(__file__).resolve().parents[2]
            / "artifacts"
            / "harnesses"
        )
        v000 = AgentClinicAdapter(
            harness_artifact=HarnessArtifact.load(
                artifact_root / "baseline" / "diagnostic_efficiency_v000.toml"
            )
        )
        v001 = AgentClinicAdapter(
            harness_artifact=HarnessArtifact.load(
                artifact_root / "ablations" / "grounded_measurement_v001.toml"
            )
        )
        config = {**self.config, "measurement_llm": "unused"}

        self.assertIs(type(v000.build_measurement(self.scenario, config)), MeasurementAgent)
        self.assertIs(
            type(v001.build_measurement(self.scenario, config)),
            GroundedMeasurementAgent,
        )


if __name__ == "__main__":
    unittest.main()
