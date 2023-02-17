import os
import os.path as osp
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from distributed_dagger.ddp import DecentralizedDistributedMixin, all_reduce, rank0_only


class DAggerBase:
    num_workers: int = 1
    is_distributed: bool = False

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
        window_size: int = 50,
        log_interval: int = 10,
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
        self.num_updates = num_updates
        self.teacher_forcing = teacher_forcing
        self.tb_dir = tb_dir
        self.checkpoint_folder = checkpoint_folder

        if updates_per_ckpt % log_interval == 0:
            updates_per_ckpt = max(updates_per_ckpt // log_interval, 1) * log_interval
            print(
                f"WARNING: updates_per_ckpt must be divisible by log_interval, setting "
                f"updates_per_ckpt to {updates_per_ckpt}"
            )
        self.updates_per_ckpt = updates_per_ckpt  # num policy updates per checkpoint
        self.log_interval = log_interval  # num policy updates per tensorboard update

        self.ckpt_id = 0
        self.last_log_time = -1
        self.num_steps_done = 0
        self.num_train_iter = 0
        self.num_updates_done = 0
        self.writer = None
        # Track scalar info stats for the last window_size episodes
        self.window_episode_stats = defaultdict(lambda: deque(maxlen=window_size))
        # Track loss for the last window_size updates
        self.loss_deque = deque(maxlen=window_size)

    @property
    def num_envs(self):
        return self.envs.num_envs

    def get_teacher_actions(self, observations):
        """Applied to output of self.transform_observations(). Must return an action and
        the teacher's recurrent hidden state (which can be None)"""
        raise NotImplementedError

    def get_student_actions(self, observations):
        """Applied to output of self.transform_observations(). Must return an action and
        the student's recurrent hidden state (which can be None)"""
        raise NotImplementedError

    def initialize_rollout(self, observations):
        raise NotImplementedError

    def update_rollout(self, observations, teacher_actions):
        raise NotImplementedError

    def get_last_obs(self):
        raise NotImplementedError

    def compute_loss(self):
        raise NotImplementedError

    def update_episode_stats(self, infos, dones):
        scalar_infos = extract_scalars_from_infos(infos)
        for k, scalar_list in scalar_infos.items():
            for idx, s in enumerate(scalar_list):
                if dones[idx]:
                    self.window_episode_stats[k].append(s)

    def log_data(self, mean_loss):
        mean_stats = {
            k: torch.tensor(np.mean(v) if len(v) > 0 else 0.0)
            for k, v in self.window_episode_stats.items()
        }
        stats_ordering = sorted(self.window_episode_stats.keys())
        mean_stats = torch.stack(
            [mean_stats[k] for k in stats_ordering] + [torch.tensor(mean_loss)], 0
        )
        if self.is_distributed:
            mean_stats = all_reduce(mean_stats, self.device)

        if not rank0_only():
            return  # only rank 0 should move on to log/tensorboard/checkpoint

        if self.is_distributed:
            mean_stats /= self.num_workers

        log_time = time.time() - self.last_log_time
        fps = (
            self.num_workers * self.num_envs * self.batch_length * self.log_interval
        ) / log_time
        self.last_log_time = time.time()
        metrics = {k: mean_stats[idx] for idx, k in enumerate(stats_ordering)}
        total_steps = self.num_steps_done * self.num_workers
        names = [("", total_steps), ("_per_update", self.num_updates_done)]
        for name, x in names:
            for k, v in metrics.items():
                self.writer.add_scalar(f"metrics{name}/{k}", v, x)
            self.writer.add_scalar(f"learner{name}/loss", mean_stats[-1], x)
            self.writer.add_scalar(f"perf{name}/fps", fps, x)

        if self.should_save_now():
            checkpoint, ckpt_path = self.generate_checkpoint()
            torch.save(checkpoint, ckpt_path)

        # Return these values for any child classes that may want to use them
        return fps, mean_stats, stats_ordering

    @rank0_only
    def setup_tensorboard(self):
        if self.tb_dir != "":
            os.makedirs(self.tb_dir, exist_ok=True)
            self.writer = SummaryWriter(self.tb_dir)
        else:
            self.writer = None

    def transform_observations(self, observations):
        """Applied to the observations output of self.sift_env_outputs()"""
        return observations

    def sift_env_outputs(self, outputs):
        """Applied to outputs of self.envs.step()"""
        observations, rewards, dones, infos = outputs
        return observations, rewards, dones, infos

    def step_envs(self, actions):
        outputs = self.envs.step(actions)
        observations, rewards, dones, infos = self.sift_env_outputs(outputs)
        observations = self.transform_observations(observations)
        return observations, rewards, dones, infos

    def _with_grad(self, actor_critic_grad_fn_name: str, *args, **kwargs):
        # Needs to be its own method for DDP inheritance
        func = getattr(self.actor_critic, actor_critic_grad_fn_name)
        return func(*args, **kwargs)

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
        self.initialize_rollout(observations)
        self.last_log_time = time.time()

    def train_loop(self):
        while self.percent_done() < 1.0:
            with torch.inference_mode():  # noqa
                for _ in range(self.batch_length):
                    observations = self.get_last_obs()
                    action_t = self.get_teacher_actions(observations)
                    action_s = self.get_student_actions(observations)
                    observations, rewards, dones, infos = self.step_envs(
                        action_t if self.teacher_forcing else action_s
                    )
                    self.update_rollout(observations, action_t)
                    self.update_episode_stats(infos, dones)
                    self.num_steps_done += self.num_envs
                    self.num_train_iter += 1
            mean_loss = self.update()
            self.num_updates_done += 1
            if self.num_updates_done % self.log_interval == 0:
                self.log_data(mean_loss)

    def update(self):
        raise NotImplementedError

    def train(self):
        self.train_setup()
        self.train_loop()


class DAggerDDP(DecentralizedDistributedMixin, DAggerBase):
    def train(self):
        if self.is_distributed:
            torch.distributed.barrier()  # noqa
        super().train()


def _extract_scalars_from_info(
    info: Dict[str, Any], blacklist: List[str]
) -> Dict[str, float]:
    result = {}
    for k, v in info.items():
        if not isinstance(k, str) or k in blacklist:
            continue

        if isinstance(v, dict):
            result.update(
                {
                    k + "." + subk: subv
                    for subk, subv in _extract_scalars_from_info(v, blacklist).items()
                    if isinstance(subk, str) and k + "." + subk not in blacklist
                }
            )
        # Things that are scalar-like will have an np.size of 1.
        # Strings also have an np.size of 1, so explicitly ban those
        elif np.size(v) == 1 and not isinstance(v, str):
            result[k] = float(v)

    return result


def extract_scalars_from_infos(
    infos: List[Dict[str, Any]], blacklist: Optional[List[str]] = None
) -> Dict[str, List[float]]:
    """Extracts scalar values from a list of info dicts. Only values corresponding to
    keys that are strings and not in the blacklist will be extracted. Values that are
    dicts will be flattened, with the keys of the dicts being concatenated with a
    period. For example, if the info dict is {"a": {"b": 1, "c": 2}, "d": 3}, then
    the result will be {"a.b": 1, "a.c": 2, "d": 3}."""
    if blacklist is None:
        blacklist = []
    results = defaultdict(list)
    for i in range(len(infos)):
        for k, v in _extract_scalars_from_info(infos[i], blacklist).items():
            results[k].append(v)

    return results
