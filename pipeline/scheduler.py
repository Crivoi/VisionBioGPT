"""
https://github.com/coastalcph/trldc/blob/main/dainlp/training/scheduler.py
"""
import logging
import math

import torch
from transformers.file_utils import ExplicitEnum

logger = logging.getLogger(__name__)

"""https://github.com/coastalcph/trldc/blob/main/dainlp/training/scheduler.py#L9"""


def get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


"""https://github.com/coastalcph/trldc/blob/main/dainlp/training/scheduler.py#L18"""


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


"""https://github.com/coastalcph/trldc/blob/main/dainlp/training/scheduler.py#L28"""


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    LINEAR_WITH_WARMUP = "linear_with_warmup"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


"""https://github.com/coastalcph/trldc/blob/main/dainlp/training/scheduler.py#L39"""

TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.LINEAR_WITH_WARMUP: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    # SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    # SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    # SchedulerType.CONSTANT: get_constant_schedule,
    # SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
}

"""https://github.com/coastalcph/trldc/blob/main/dainlp/training/scheduler.py#L52"""


def create_scheduler(lr_scheduler_type, optimizer, num_training_steps, warmup_steps=0, warmup_ratio=0):
    lr_scheduler_type = SchedulerType(lr_scheduler_type)
    lr_scheduler_func = TYPE_TO_SCHEDULER_FUNCTION[lr_scheduler_type]
    if lr_scheduler_type == SchedulerType.CONSTANT: return lr_scheduler_func(optimizer)

    if warmup_steps == 0: warmup_steps = math.ceil(num_training_steps * warmup_ratio)
    if lr_scheduler_type == SchedulerType.CONSTANT_WITH_WARMUP:
        return lr_scheduler_func(optimizer, num_warmup_steps=warmup_steps)
    return lr_scheduler_func(optimizer, warmup_steps, num_training_steps)
