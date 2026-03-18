from __future__ import annotations

from src.factors.base import Factor
from src.factors.momentum import MomentumFactor
from src.factors.value import EVToEBITDA, PBRatio, PERatio

_REGISTRY = {}


def register(factor_class_or_factory, name=None):
    """Register a factor by name."""
    if isinstance(factor_class_or_factory, type):
        instance = factor_class_or_factory()
        _REGISTRY[name or instance.name] = instance
    else:
        instance = factor_class_or_factory
        _REGISTRY[name or instance.name] = instance


def get_factor(name) -> Factor:
    """Look up a factor by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown factor: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_factors():
    """List all registered factor names."""
    return list(_REGISTRY.keys())


# Register built-in factors
register(PERatio)
register(PBRatio)
register(EVToEBITDA)
register(MomentumFactor(63, name="momentum_3m"), name="momentum_3m")
register(MomentumFactor(126, name="momentum_6m"), name="momentum_6m")
register(MomentumFactor(252, name="momentum_12m"), name="momentum_12m")
