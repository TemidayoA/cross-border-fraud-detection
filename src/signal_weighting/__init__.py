"""
Signal Weighting Module

Dynamic weight adjustment for corridor-aware fraud detection.
"""

from .dynamic_weights import (
    DynamicWeightCalculator,
    FraudScorer,
    CorridorProfile,
    CorridorMultipliers,
    BaseWeights,
    CORRIDOR_MULTIPLIERS,
    learn_corridor_multipliers,
)

__all__ = [
    'DynamicWeightCalculator',
    'FraudScorer',
    'CorridorProfile',
    'CorridorMultipliers',
    'BaseWeights',
    'CORRIDOR_MULTIPLIERS',
    'learn_corridor_multipliers',
]
