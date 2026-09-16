"""Deprecated compatibility shim; use ``lumina_dimoo.objectives.gce``."""

from lumina_dimoo.objectives.gce import GCEObjective, GroupedCrossEntropyLoss

__all__ = ["GCEObjective", "GroupedCrossEntropyLoss"]
