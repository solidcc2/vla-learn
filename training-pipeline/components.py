"""Dynamic construction of trusted training components."""

from collections.abc import Iterable
from importlib import import_module
import inspect
from typing import Any

import torch
from torch import nn

from data_modules import DataModule


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


def _construct(target: str, params: dict, *args) -> Any:
    constructor = resolve_target(target)
    if not isinstance(constructor, type):
        raise TypeError(f"Target {target!r} is not a class")
    try:
        return constructor(*args, **params)
    except Exception as error:
        raise TypeError(f"Could not construct target {target!r}: {error}") from error


def create_model(config: dict) -> nn.Module:
    target = config["model"]["target"]
    model = _construct(target, config["model"]["params"])
    if not isinstance(model, nn.Module):
        raise TypeError(f"Target {target!r} did not create an nn.Module")
    return model


def create_data_module(config: dict) -> DataModule:
    target = config["data"]["target"]
    data_module = _construct(target, config["data"]["params"])
    if not isinstance(data_module, DataModule):
        raise TypeError(
            f"Target {target!r} did not create a DataModule"
        )
    return data_module


def create_optimizer(
    config: dict,
    parameters: Iterable[nn.Parameter],
) -> torch.optim.Optimizer:
    target = config["optimizer"]["target"]
    optimizer_type = resolve_target(target)
    if not isinstance(optimizer_type, type) or not issubclass(optimizer_type, torch.optim.Optimizer):
        raise TypeError(f"Target {target!r} is not an Optimizer class")
    try:
        return optimizer_type(parameters, **config["optimizer"]["params"])
    except Exception as error:
        raise TypeError(f"Could not construct target {target!r}: {error}") from error


def create_scheduler(
    config: dict,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config["scheduler"] is None:
        return None
    target = config["scheduler"]["target"]
    scheduler_type = resolve_target(target)
    base_type = torch.optim.lr_scheduler.LRScheduler
    if not isinstance(scheduler_type, type) or not issubclass(scheduler_type, base_type):
        raise TypeError(f"Target {target!r} is not an LRScheduler class")
    if scheduler_type is torch.optim.lr_scheduler.SequentialLR:
        raise TypeError("SequentialLR is not supported by the epoch scheduler configuration")
    try:
        scheduler = scheduler_type(optimizer, **config["scheduler"]["params"])
    except Exception as error:
        raise TypeError(f"Could not construct target {target!r}: {error}") from error
    required = [
        parameter
        for parameter in inspect.signature(scheduler.step).parameters.values()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ) and parameter.default is inspect.Parameter.empty
    ]
    if required:
        raise TypeError(
            f"Scheduler {target!r} requires arguments to step(); "
            "only epoch schedulers with step() are supported"
        )
    return scheduler
