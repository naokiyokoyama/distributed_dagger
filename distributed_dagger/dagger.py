import os
import os.path as osp
from typing import Any

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from distributed_dagger.distributed_dagger.ddp.ddp import DecentralizedDistributedMixin


class DAggerBase:
    def __init__(
        self,
        envs: Any,
        actor_critic: torch.nn.Module,
        optimizer: optim.Optimizer,
        batch_length: int,
        total_num_steps: int,
        device: torch.device,
        updates_per_ckpt: int = 100,
        num_updates: int = -1,
        teacher_forcing: bool = False,
        tb_dir: str = "",
        checkpoint_folder: str = "",
    ):
        self.envs = envs
        self.actor_critic = actor_critic
        self.optimizer = optimizer
        self.batch_length = batch_length
        self.total_num_steps = total_num_steps
        self.device = device
        self.updates_per_ckpt = updates_per_ckpt
        self.num_updates = num_updates
        self.teacher_forcing = teacher_forcing
        self.tb_dir = tb_dir
        self.checkpoint_folder = checkpoint_folder

        self.ckpt_id = 0
        self.num_steps_done = 0
        self.num_train_iter = 0
        self.num_updates_done = 0
        self.writer = None

    @property
    def num_envs(self):
        return self.envs.num_envs

    def get_teacher_actions(self, observations):
        """Applied to output of self.transform_observations(), sent to
        self.action_loss()"""
        raise NotImplementedError

    def get_student_actions(self, observations):
        """Applied to output of self.transform_observations(), sent to
        self.envs.step()"""
        raise NotImplementedError

    def update_metrics(self, observations, rewards, dones, infos, action_loss):
        """Print, log, tensorboard, or otherwise record metrics for this iteration"""
        raise NotImplementedError

    def setup_tensorboard(self):
        if self.tb_dir != "":
            os.makedirs(self.tb_dir, exist_ok=True)
            self.writer = SummaryWriter(self.tb_dir)
        else:
            self.writer = None

    def transform_observations(self, observations):
        """Applied to the observations output of self.envs.step()"""
        return observations

    def sift_env_outputs(self, outputs):
        """Applied to all outputs of self.envs.step()"""
        observations, rewards, dones, infos = outputs
        return observations, rewards, dones, infos

    def step_envs(self, actions):
        outputs = self.envs.step(actions)
        observations, rewards, dones, infos = self.sift_env_outputs(outputs)
        observations = self.transform_observations(observations)
        return observations, rewards, dones, infos

    def _compute_log_probs(self, actions):
        # Needs to be its own method for DDP inheritance
        return self.actor_critic.compute_log_probs(actions)

    def action_loss(self, teacher_actions):
        log_probs = self._compute_log_probs(teacher_actions)
        return -log_probs.mean()

    def generate_checkpoint(self):
        checkpoint = {
            "state_dict": self.actor_critic.state_dict(),
            "iteration": self.num_train_iter,
            "batch": self.num_updates_done,
        }
        filename = f"ckpt.{self.ckpt_id}_{self.num_updates_done}.pth"
        ckpt_path = osp.join(self.checkpoint_folder, filename)
        self.ckpt_id += 1
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)
        return checkpoint, ckpt_path

    def percent_done(self) -> float:
        if self.num_updates != -1:
            return self.num_updates_done / self.num_updates
        else:
            return self.num_steps_done / self.total_num_steps

    def should_save_now(self):
        return (
            self.num_updates_done % self.updates_per_ckpt == 0
            and self.checkpoint_folder != ""
        )

    def train_setup(self):
        self.setup_tensorboard()
        observations = self.envs.reset()
        observations = self.transform_observations(observations)
        return observations

    def train_loop(self, observations):
        action_loss = None
        while self.percent_done() < 1.0:
            teacher_actions = self.get_teacher_actions(observations)
            student_actions = self.get_student_actions(observations)
            actions = teacher_actions if self.teacher_forcing else student_actions

            curr_loss = self.action_loss(teacher_actions)
            action_loss = curr_loss if action_loss is None else action_loss + curr_loss

            observations, rewards, dones, infos = self.step_envs(actions)

            if self.num_train_iter % self.batch_length == 0:
                self.optimizer.zero_grad()
                action_loss = action_loss.mean() / self.batch_length
                action_loss.backward()
                self.optimizer.step()

                self.num_updates_done += 1
                action_loss = None

            self.num_steps_done += self.num_envs
            self.num_train_iter += 1

            self.update_metrics(observations, rewards, dones, infos, action_loss)

    def train(self):
        observations = self.train_setup()
        self.train_loop(observations)


class DAggerDDP(DecentralizedDistributedMixin, DAggerBase):
    def train(self):
        if self._is_distributed:
            torch.distributed.barrier()
        super().train()
