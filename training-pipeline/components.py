"""Dynamic construction of trusted training components."""

from collections.abc import Iterable
from importlib import import_module
import inspect
from typing import Any

import torch
from torch import nn


def resolve_target(target: str) -> Any:
    """Resolve module.path:AttributeName to a Python object."""
    module_name, separator, attribute_name = target.partition(":")
    if not separator or not module_name or not attribute_name or ":" in attribute_name:
        raise ValueError(f"Invalid target {target!r}; expected 'module.path:AttributeName'")
    try:
        module = import_module(module_name)
    except (ImportError, ValueError) as error:
        raise ValueError(f"Could not import target module for {target!r}") from error
    try:
        return getattr(module, attribute_name)
    except AttributeError as error:
        raise ValueError(f"Target {target!r} does not exist") from error


def _construct(spec: dict, *args) -> Any:
    target = spec["target"]
    constructor = resolve_target(target)
    if not isinstance(constructor, type):
        raise TypeError(f"Target {target!r} is not a class")
    try:
        return constructor(*args, **spec["params"])
    except Exception as error:
        raise TypeError(f"Could not construct target {target!r}: {error}") from error


def create_model(spec: dict) -> nn.Module:
    model = _construct(spec)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Target {spec['target']!r} did not create an nn.Module")
    return model


def create_optimizer(
    spec: dict,
    parameters: Iterable[nn.Parameter],
) -> torch.optim.Optimizer:
    optimizer_type = resolve_target(spec["target"])
    if not isinstance(optimizer_type, type) or not issubclass(optimizer_type, torch.optim.Optimizer):
        raise TypeError(f"Target {spec['target']!r} is not an Optimizer class")
    try:
        return optimizer_type(parameters, **spec["params"])
    except Exception as error:
        raise TypeError(f"Could not construct target {spec['target']!r}: {error}") from error


def create_scheduler(
    spec: dict | None,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if spec is None:
        return None
    scheduler_type = resolve_target(spec["target"])
    base_type = torch.optim.lr_scheduler.LRScheduler
    if not isinstance(scheduler_type, type) or not issubclass(scheduler_type, base_type):
        raise TypeError(f"Target {spec['target']!r} is not an LRScheduler class")
    if scheduler_type is torch.optim.lr_scheduler.SequentialLR:
        raise TypeError("SequentialLR is not supported by the epoch scheduler configuration")
    try:
        scheduler = scheduler_type(optimizer, **spec["params"])
    except Exception as error:
        raise TypeError(f"Could not construct target {spec['target']!r}: {error}") from error
    required = [
        parameter
        for parameter in inspect.signature(scheduler.step).parameters.values()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ) and parameter.default is inspect.Parameter.empty
    ]
    if required:
        raise TypeError(
            f"Scheduler {spec['target']!r} requires arguments to step(); "
            "only epoch schedulers with step() are supported"
        )
    return scheduler
