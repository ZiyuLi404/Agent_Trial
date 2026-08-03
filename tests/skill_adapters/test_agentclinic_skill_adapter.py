import tempfile
import unittest
from pathlib import Path

from AgentClinic.agentclinic import DoctorAgent
from agent_system_adapters.agentclinic import AgentClinicAdapter
from agent_system_adapters.agentclinic.skill_doctor import SkillDoctorAgent
from change_generators.harnesses import HarnessArtifact
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

    def test_harness_changes_orchestration_without_changing_core(self):
        harness_path = (
            Path(__file__).resolve().parents[2]
            / "change_generators"
            / "harnesses"
            / "artifacts"
            / "agentclinic"
            / "diagnostic_efficiency"
            / "v000.toml"
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


if __name__ == "__main__":
    unittest.main()
