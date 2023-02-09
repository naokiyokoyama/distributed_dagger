import os
import os.path as osp
from typing import Any

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter


class DaggerBase:
    def __init__(
        self,
        envs: Any,
        actor_critic: torch.nn.Module,
        optimizer: optim.Optimizer,
        batch_length: int,
        total_num_steps: int,
        device: torch.device,
        lr: float = 3e-4,
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
        self.lr = lr
        self.updates_per_ckpt = updates_per_ckpt
        self.num_updates = num_updates
        self.teacher_forcing = teacher_forcing
        self.tb_dir = tb_dir
        self.checkpoint_folder = checkpoint_folder

        self.writer = None
        self.ckpt_id = 0
        self.num_updates_done = 0
        self.num_steps_done = 0
        self.num_train_iter = 0

    @property
    def num_envs(self):
        return self.envs.num_envs

    def get_model_params(self):
        return self.actor_critic.parameters()

    def setup_tensorboard(self):
        if self.tb_dir != "":
            print(f"Creating tensorboard at {self.tb_dir}...")
            os.makedirs(self.tb_dir, exist_ok=True)
            self.writer = SummaryWriter(self.tb_dir)
        else:
            self.writer = None

    def transform_observations(self, observations):
        return observations

    def sift_env_outputs(self, outputs):
        observations, rewards, dones, infos = outputs
        observations = self.transform_observations(observations)
        return observations, rewards, dones, infos

    def step_envs(self, actions):
        outputs = self.envs.step(actions)
        observations, rewards, dones, infos = self.sift_env_outputs(outputs)
        return observations, rewards, dones, infos

    def get_teacher_actions(self, observations):
        raise NotImplementedError

    def action_loss(self, teacher_actions, student_actions):
        raise NotImplementedError

    def update_tensorboard(self, observations, rewards, dones, infos, action_loss):
        raise NotImplementedError

    def get_checkpoint(self, iteration, batch_num):
        checkpoint = {
            "state_dict": self.actor_critic.state_dict(),
            "iteration": iteration,
            "batch": batch_num,
        }
        filename = f"ckpt.{self.ckpt_id}_{batch_num}.pth"
        ckpt_path = osp.join(self.checkpoint_folder, filename)
        self.ckpt_id += 1
        return checkpoint, ckpt_path

    def percent_done(self) -> float:
        if self.num_updates != -1:
            return self.num_updates_done / self.num_updates
        else:
            return self.num_steps_done / self.total_num_steps

    def train(self):
        self.setup_tensorboard()
        observations = self.envs.reset()
        observations = self.transform_observations(observations)

        action_loss = torch.tensor(0, device=self.device)
        while self.percent_done() < 1.0:
            teacher_actions = self.get_teacher_actions(observations)
            student_actions = self.actor_critic(observations)
            actions = teacher_actions if self.teacher_forcing else student_actions

            action_loss += self.action_loss(actions, teacher_actions)

            observations, rewards, dones, infos = self.step_envs(actions)

            self.update_tensorboard(observations, rewards, dones, infos, action_loss)

            if self.num_train_iter % self.batch_length == 0:
                self.optimizer.zero_grad()
                action_loss = torch.mean(action_loss) / self.batch_length
                action_loss.backward()
                self.optimizer.step()

                self.num_updates_done += 1
                action_loss = torch.tensor(0, device=self.device)

                if (
                    self.num_updates_done % self.updates_per_ckpt == 0
                    and self.checkpoint_folder != ""
                ):
                    # Save checkpoint
                    ckpt, ckpt_path = self.get_checkpoint(
                        self.num_train_iter, self.num_updates_done
                    )
                    torch.save(ckpt, ckpt_path)
                    print("Saved checkpoint:", ckpt_path)

            self.num_steps_done += self.num_envs
            self.num_train_iter += 1
