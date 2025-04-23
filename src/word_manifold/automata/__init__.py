"""Automata package for word manifold system."""

from .cellular_rules import (
    CellularRule,
    RuleSequence,
    create_predefined_rules,
    create_predefined_sequences,
    HermeticPrinciple,
    ElementalForce,
    VibrationDirection,
    RuleParameterSet
)

from .system import (
    AutomataSystem,
    EvolutionPattern
)

__all__ = [
    'CellularRule',
    'RuleSequence',
    'create_predefined_rules',
    'create_predefined_sequences',
    'HermeticPrinciple',
    'ElementalForce',
    'VibrationDirection',
    'RuleParameterSet',
    'AutomataSystem',
    'EvolutionPattern'
]
