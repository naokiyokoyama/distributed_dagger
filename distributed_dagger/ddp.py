#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import functools
from typing import Callable, Optional, Tuple, Union, overload

import numpy as np
import torch
from torch import distributed as distrib


def _recursive_apply(inp, fn):
    if isinstance(inp, dict):
        return type(inp)((k, _recursive_apply(v, fn)) for k, v in inp.items())
    elif isinstance(inp, (tuple, list)):
        return type(inp)(_recursive_apply(v, fn) for v in inp)
    else:
        return fn(inp)


def _cpu_to_numpy(inp):
    return _recursive_apply(inp, lambda t: t.numpy() if t.device.type == "cpu" else t)


def _numpy_to_cpu(inp):
    return _recursive_apply(
        inp,
        lambda t: torch.from_numpy(t) if isinstance(t, np.ndarray) else t,
    )


def distributed_var_mean(
    values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes the mean and variances of a tensor over multiple workers.

    This method is equivalent to first collecting all versions of values and
    then computing the mean and variance locally over that

    :param values: (*,) shaped tensors to compute mean and variance over.  Assumed
                        to be solely the workers local copy of this tensor,
                        the resultant mean and variance will be computed
                        over _all_ workers version of this tensor.
    """
    assert distrib.is_initialized(), "Distributed must be initialized"

    world_size = distrib.get_world_size()

    mean = values.mean()
    distrib.all_reduce(mean)
    mean = mean / world_size

    var = (values - mean).pow(2).mean()
    distrib.all_reduce(var)
    var = var / world_size

    return var, mean


def all_reduce(t: torch.Tensor, device=torch.device) -> torch.Tensor:
    r"""All reduce helper method that moves things to the correct
    device and only runs if distributed
    """
    orig_device = t.device
    t = t.to(device=device)
    torch.distributed.all_reduce(t)

    return t.to(device=orig_device)


@overload
def rank0_only() -> bool:
    ...


@overload
def rank0_only(fn: Callable) -> Callable:
    ...


def rank0_only(fn: Optional[Callable] = None) -> Union[Callable, bool]:
    r"""Helper function to only execute code if a process is world rank 0

    Can be used both as a function in an if statement,

    .. code:: py

        if rank0_only():
            ...

    or as a decorator,

    .. code:: py

        @rank0_only
        def fn_for_r0_only(...):
            ...

    :param fn: Function to wrap and only execute if the process is rank 0.
        If a process is rank 0, the function will be run and it's return value
        will be returned.  If a process is not rank 0, then the function will not
        be ran and :py:`None` will be returned.

    :return: The wrapped function if :p:`fn` is not :py:`None`, otherwise
        whether or not this process is rank 0
    """
    if fn is None:
        return (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        )

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if rank0_only():
            return fn(*args, **kwargs)
        return None

    return _wrapper


class _GradFnWrapper(torch.nn.Module):
    r"""Wrapper on compute_log_probs that allows that to be called from forward.
    This is needed to interface with DistributedDataParallel's forward call
    """

    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, actor_critic_grad_fn_name, *args, **kwargs):
        # We then convert numpy arrays back to a CPU tensor.
        # This is needed for older versions of pytorch that haven't deprecated
        # the single-process multi-device version of DDP
        func = getattr(self.actor_critic, actor_critic_grad_fn_name)
        return func(*_numpy_to_cpu(args), **_numpy_to_cpu(kwargs))


class DecentralizedDistributedMixin:
    is_distributed: bool = False
    actor_critic: torch.nn.Module
    device: torch.device

    @staticmethod
    def _compute_var_mean(x):
        return distributed_var_mean(x)

    def init_distributed(self, find_unused_params: bool = True) -> None:
        r"""Initializes distributed training for the model

        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model

        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where
                                   unused in the forward pass, otherwise the gradient
                                   reduction will not work correctly.
        """
        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:  # noqa: SIM119
            def __init__(self, model, device):
                if device.type == "cuda":
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model,
                        device_ids=[device],
                        output_device=device,
                        find_unused_parameters=find_unused_params,
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model,
                        find_unused_parameters=find_unused_params,
                    )

        self._with_grad_wrapper = Guard(_GradFnWrapper(self.actor_critic), self.device)
        self.is_distributed = True

    def _with_grad(self, actor_critic_grad_fn_name, *args, **kwargs):
        r"""Internal method that calls Policy.compute_log_probs.  This is used instead
        of calling that directly so that that call can be overrided with inheritance
        """
        if not self.is_distributed:
            return super()._with_grad(
                actor_critic_grad_fn_name, *args, **kwargs
            )  # noqa
        # DistributedDataParallel moves all tensors to the device (or devices)
        # So we need to make anything that is on the CPU into a numpy array
        # This is needed for older versions of pytorch that haven't deprecated
        # the single-process multi-device version of DDP
        return self._with_grad_wrapper.ddp(
            actor_critic_grad_fn_name, *_cpu_to_numpy(args), **_cpu_to_numpy(kwargs)
        )
