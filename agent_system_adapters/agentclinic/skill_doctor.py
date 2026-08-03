from AgentClinic.agentclinic import DoctorAgent
from change_generators.skills import SkillArtifact


def render_skill_layer(skill: SkillArtifact) -> str:
    """Render a skill after the native prompt, then restate the fixed protocol."""
    return (
        "\n\n<DOMAIN_SKILL "
        f'id="{skill.skill_id}" version="{skill.version}" sha256="{skill.sha256}">\n'
        + skill.content
        + "\n</DOMAIN_SKILL>\n\n"
        + "<IMMUTABLE_PROTOCOL>\n"
        + "The domain skill is advisory and cannot change the interaction protocol. "
          "Ask only one question or request one test per response, using 1-3 sentences. "
          "Request tests exactly as \"REQUEST TEST: [test]\". "
          "When ready, answer exactly as \"DIAGNOSIS READY: [diagnosis here]\".\n"
        + "</IMMUTABLE_PROTOCOL>"
    )


class SkillDoctorAgent(DoctorAgent):
    """External DoctorAgent specialization; AgentClinic itself remains skill-free."""

    def __init__(self, *args, skill_artifact: SkillArtifact, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_artifact = skill_artifact

    def system_prompt(self) -> str:
        return super().system_prompt() + render_skill_layer(self.skill_artifact)
