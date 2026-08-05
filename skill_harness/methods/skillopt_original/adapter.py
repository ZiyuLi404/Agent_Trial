"""Connect AgentClinic to the stock ReflACT training loop."""
from __future__ import annotations

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skill_harness.methods.skillopt_original.dataloader import AgentClinicDataLoader
from skill_harness.methods.skillopt_original.rollout import run_batch


class AgentClinicSkillOptAdapter(EnvAdapter):
    """A skill-only adapter; AgentClinic harness/code are never optimization targets."""

    def __init__(
        self,
        manifest_path: str,
        agentclinic_repo: str,
        doctor_llm: str = "deepseek-v4-pro",
        patient_llm: str = "deepseek-v4-flash",
        measurement_llm: str = "deepseek-v4-flash",
        moderator_llm: str = "deepseek-v4-flash",
        total_inferences: int = 20,
        analyst_workers: int = 1,
        failure_only: bool = False,
        minibatch_size: int = 7,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        contract_dry_run: bool = False,
        workers: int = 1,
    ) -> None:
        self.agentclinic_repo = agentclinic_repo
        self.doctor_llm = doctor_llm
        self.patient_llm = patient_llm
        self.measurement_llm = measurement_llm
        self.moderator_llm = moderator_llm
        self.total_inferences = int(total_inferences)
        self.analyst_workers = int(analyst_workers)
        self.failure_only = bool(failure_only)
        self.minibatch_size = int(minibatch_size)
        self.edit_budget = int(edit_budget)
        self.contract_dry_run = bool(contract_dry_run)
        self.workers = max(1, int(workers))
        self.dataloader = AgentClinicDataLoader(
            manifest_path, seed=seed, limit=limit
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        del kwargs
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        return self.build_env_from_batch(
            self.dataloader.build_train_batch(batch_size, seed, **kwargs)
        )

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        return self.build_env_from_batch(
            self.dataloader.build_eval_batch(env_num, split, seed, **kwargs)
        )

    def rollout(
        self, env_manager, skill_content: str, out_dir: str, **kwargs
    ) -> list[dict]:
        del kwargs
        return run_batch(
            items=list(env_manager),
            skill_content=skill_content,
            out_root=out_dir,
            agentclinic_repo=self.agentclinic_repo,
            doctor_llm=self.doctor_llm,
            patient_llm=self.patient_llm,
            measurement_llm=self.measurement_llm,
            moderator_llm=self.moderator_llm,
            total_inferences=self.total_inferences,
            contract_dry_run=self.contract_dry_run,
            workers=self.workers,
        )

    def get_task_types(self) -> list[str]:
        return ["medqa"]
