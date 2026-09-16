"""Compatibility import for the GCE objective now owned by ``objectives``."""

from objectives.gce import GroupedCrossEntropyLoss

__all__ = ["GroupedCrossEntropyLoss"]
