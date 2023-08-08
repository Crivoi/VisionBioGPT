import logging
import torch

from torch.optim import SGD
from transformers.file_utils import ExplicitEnum
from transformers.optimization import AdamW

logger = logging.getLogger(__name__)

"""https://github.com/coastalcph/trldc/blob/main/dainlp/training/optimizer.py#L11"""


class OptimizerNames(ExplicitEnum):
    ADAMW_HF = "adamw_hf"
    SGD = "sgd"


"""https://github.com/coastalcph/trldc/blob/main/dainlp/training/optimizer.py#L17"""


def get_parameter_names(model, skipped_types):
    result = []
    for name, child in model.named_children():
        result += [f"{name}.{n}" for n in get_parameter_names(child, skipped_types) if
                   not isinstance(child, tuple(skipped_types))]
    result += list(model._parameters.keys())
    return result


"""https://github.com/coastalcph/trldc/blob/main/dainlp/training/optimizer.py#L27"""


def get_optimizer_cls_and_kwargs(args):
    optimizer_kwargs = {"lr": args.learning_rate}
    if args.optim == OptimizerNames.ADAMW_HF:
        adam_kwargs = {"betas": (0.9, 0.999), "eps": 1e-8}
        optimizer_cls = AdamW
        optimizer_kwargs.update(adam_kwargs)
    elif args.optim == OptimizerNames.SGD:
        sgd_kwargs = {"momentum": 0.9}
        optimizer_cls = SGD
        optimizer_kwargs.update(sgd_kwargs)
    else:
        raise ValueError(args.optim)
    return optimizer_cls, optimizer_kwargs


"""https://github.com/coastalcph/trldc/blob/main/dainlp/training/optimizer.py#L43"""


def create_optimizer(model, args):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [n for n in decay_parameters if "bias" not in n]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if n in decay_parameters],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if n not in decay_parameters], "weight_decay": 0.0}]
    optimizer_cls, optimizer_kwargs = get_optimizer_cls_and_kwargs(args)
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer
