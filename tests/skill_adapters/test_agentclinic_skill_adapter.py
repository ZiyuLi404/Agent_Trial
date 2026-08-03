import tempfile
import unittest
from pathlib import Path

from AgentClinic.agentclinic import DoctorAgent
from agent_system_adapters.agentclinic import AgentClinicAdapter
from agent_system_adapters.agentclinic.skill_doctor import SkillDoctorAgent
from change_generators.skills import SkillArtifact


class DummyScenario:
    def examiner_information(self):
        return "A patient with fatigue."


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


if __name__ == "__main__":
    unittest.main()
