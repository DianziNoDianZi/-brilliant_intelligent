"""Uncertainty decomposition for curiosity-driven exploration.

Phase 1: IdentityDecomposer — all error is epistemic.
Phase 1.5: EnsembleDecomposer — disagreement across ensemble members.
Phase 2+: More sophisticated methods (e.g., information gain).
"""

import numpy as np
from abc import ABC, abstractmethod


class UncertaintyDecomposer(ABC):
    """Abstract interface for decomposing uncertainty.

    Subclasses must implement decompose() returning:
        'epistemic': uncertainty due to lack of knowledge (should reward)
        'aleatoric': uncertainty due to environment noise (should not reward)
        'intrinsic_reward': final scalar reward signal
    """

    @abstractmethod
    def decompose(self, **kwargs):
        pass


class IdentityDecomposer(UncertaintyDecomposer):
    """All prediction error → epistemic (Phase 1 default)."""

    def decompose(self, prediction_error, **kwargs):
        return {
            'epistemic': prediction_error,
            'aleatoric': 0.0,
            'intrinsic_reward': prediction_error,
        }


class EnsembleDecomposer(UncertaintyDecomposer):
    """Ensemble disagreement → epistemic uncertainty.

    Uses the variance across ensemble world model predictions
    as the curiosity signal. High variance means the models
    disagree → knowledge gap → explore.

    This naturally distinguishes:
    - Epistemic: members disagree (different predictions)
    - Aleatoric: members agree but are wrong (irreducible noise)
    """

    def __init__(self, scale=1.0):
        self.scale = scale
        self._uncertainty_type = 'epistemic'

    @property
    def uncertainty_type(self):
        return self._uncertainty_type

    def decompose(self, ensemble_variance=None, **kwargs):
        """ensemble_variance: scalar, mean variance across all members."""
        epistemic = float(ensemble_variance) if ensemble_variance is not None else 0.0
        return {
            'epistemic': epistemic,
            'aleatoric': 0.0,
            'intrinsic_reward': epistemic * self.scale,
        }


class IntrinsicReward:
    """Wraps a decomposer and provides a simple compute() interface.

    The sliding window normalization from Phase 1 is removed —
    raw decomposition output is used directly (proven more effective).
    """

    def __init__(self, decomposer=None):
        self.decomposer = decomposer or IdentityDecomposer()

    @property
    def uncertainty_type(self):
        if hasattr(self.decomposer, 'uncertainty_type'):
            return self.decomposer.uncertainty_type
        return 'epistemic'

    def compute(self, **kwargs):
        components = self.decomposer.decompose(**kwargs)
        return components['intrinsic_reward'], components
