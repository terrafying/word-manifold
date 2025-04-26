"""Z3 solver integration for semantic manifold analysis."""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import z3
import numpy as np
from word_manifold.manifold.pattern_selection import PatternSelector, PatternSource

@dataclass
class Z3Constraint:
    """Represents a Z3 constraint with metadata."""
    constraint: z3.ExprRef
    priority: float = 1.0
    description: str = ""
    source: str = ""

class Z3SolverManager:
    """Manages Z3 solver operations with modern features."""
    
    def __init__(
        self,
        pattern_source: PatternSource = PatternSource.HERMETIC,
        timeout: int = 30000,  # 30 seconds
        max_memory: int = 1024 * 1024 * 1024  # 1GB
    ):
        """
        Initialize the solver manager.
        
        Args:
            pattern_source: Source of patterns for selection
            timeout: Solver timeout in milliseconds
            max_memory: Maximum memory usage in bytes
        """
        self.pattern_selector = PatternSelector(pattern_source)
        self.timeout = timeout
        self.max_memory = max_memory
        self.solver = z3.Solver()
        self.solver.set("timeout", timeout)
        self.solver.set("memory_high_watermark", max_memory)
        
    def add_constraint(self, constraint: Z3Constraint):
        """
        Add a constraint to the solver.
        
        Args:
            constraint: Constraint to add
        """
        self.solver.add(constraint.constraint)
        
    def add_constraints(self, constraints: List[Z3Constraint]):
        """
        Add multiple constraints to the solver.
        
        Args:
            constraints: List of constraints to add
        """
        for constraint in constraints:
            self.add_constraint(constraint)
            
    def solve(self) -> Optional[Dict[str, Any]]:
        """
        Solve the current set of constraints.
        
        Returns:
            Solution if found, None otherwise
        """
        result = self.solver.check()
        
        if result == z3.sat:
            model = self.solver.model()
            return {
                "status": "sat",
                "model": model,
                "statistics": self.solver.statistics()
            }
        elif result == z3.unsat:
            return {
                "status": "unsat",
                "core": self.solver.unsat_core(),
                "statistics": self.solver.statistics()
            }
        else:
            return {
                "status": "unknown",
                "statistics": self.solver.statistics()
            }
            
    def reset(self):
        """Reset the solver state."""
        self.solver.reset()
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get solver statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.solver.statistics()
        return {
            "time": stats.get("time", 0),
            "memory": stats.get("memory", 0),
            "decisions": stats.get("decisions", 0),
            "propagations": stats.get("propagations", 0)
        }
        
    def optimize(self, objective: z3.ExprRef, maximize: bool = True) -> Optional[Dict[str, Any]]:
        """
        Optimize an objective function.
        
        Args:
            objective: Objective function to optimize
            maximize: Whether to maximize (True) or minimize (False)
            
        Returns:
            Optimal solution if found, None otherwise
        """
        opt = z3.Optimize()
        opt.add(self.solver.assertions())
        
        if maximize:
            opt.maximize(objective)
        else:
            opt.minimize(objective)
            
        result = opt.check()
        
        if result == z3.sat:
            return {
                "status": "sat",
                "model": opt.model(),
                "value": opt.upper(objective),
                "statistics": opt.statistics()
            }
        else:
            return {
                "status": "unsat" if result == z3.unsat else "unknown",
                "statistics": opt.statistics()
            }
            
    def get_unsat_core(self) -> List[Z3Constraint]:
        """
        Get the unsatisfiable core of constraints.
        
        Returns:
            List of constraints in the unsat core
        """
        if self.solver.check() == z3.unsat:
            core = self.solver.unsat_core()
            return [c for c in self.solver.assertions() if c in core]
        return []
        
    def add_pattern_constraints(self, pattern_name: str) -> List[Z3Constraint]:
        """
        Add constraints based on a pattern.
        
        Args:
            pattern_name: Name of the pattern to use
            
        Returns:
            List of added constraints
        """
        pattern = self.pattern_selector.get_pattern(pattern_name)
        if not pattern:
            return []
            
        # Convert pattern to Z3 constraints
        constraints = []
        for i, value in enumerate(pattern):
            var = z3.Real(f"pattern_{i}")
            constraints.append(Z3Constraint(
                constraint=var == value,
                description=f"Pattern {pattern_name} constraint {i}",
                source="pattern"
            ))
            
        self.add_constraints(constraints)
        return constraints 