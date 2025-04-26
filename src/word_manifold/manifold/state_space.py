from typing import Dict, List, Optional, Tuple
from z3 import *
import numpy as np

class StateSpaceAnalyzer:
    """A class for analyzing state spaces using Z3."""
    
    def __init__(self):
        """Initialize the StateSpaceAnalyzer."""
        self.solver = Solver()
        self.variables = {}
        
    def add_constraint(self, constraint: BoolRef):
        """
        Add a constraint to the state space.
        
        Args:
            constraint: Z3 boolean expression representing a constraint
        """
        self.solver.add(constraint)
        
    def add_word_constraint(self, word1: str, word2: str, relation: str):
        """
        Add a constraint between two words.
        
        Args:
            word1: First word
            word2: Second word
            relation: Relation type ("similar", "different", "related")
        """
        if word1 not in self.variables:
            self.variables[word1] = Real(word1)
        if word2 not in self.variables:
            self.variables[word2] = Real(word2)
            
        if relation == "similar":
            self.solver.add(self.variables[word1] == self.variables[word2])
        elif relation == "different":
            self.solver.add(self.variables[word1] != self.variables[word2])
        elif relation == "related":
            self.solver.add(self.variables[word1] > 0)
            self.solver.add(self.variables[word2] > 0)
            
    def analyze(self) -> Dict:
        """
        Analyze the state space.
        
        Returns:
            Dictionary containing analysis results
        """
        result = {
            "satisfiable": self.solver.check() == sat,
            "model": None,
            "constraints": len(self.solver.assertions()),
            "variables": list(self.variables.keys())
        }
        
        if result["satisfiable"]:
            model = self.solver.model()
            result["model"] = {
                var: model[val].as_long() if model[val].is_int() else model[val].as_fraction()
                for var, val in self.variables.items()
            }
            
        return result
    
    def find_all_models(self, max_models: int = 10) -> List[Dict]:
        """
        Find all satisfying models up to a maximum number.
        
        Args:
            max_models: Maximum number of models to find
            
        Returns:
            List of dictionaries containing model assignments
        """
        models = []
        while len(models) < max_models:
            if self.solver.check() != sat:
                break
                
            model = self.solver.model()
            model_dict = {
                var: model[val].as_long() if model[val].is_int() else model[val].as_fraction()
                for var, val in self.variables.items()
            }
            models.append(model_dict)
            
            # Add constraint to exclude this model
            self.solver.add(Or([
                self.variables[var] != val
                for var, val in model_dict.items()
            ]))
            
        return models
    
    def verify_property(self, property_expr: BoolRef) -> bool:
        """
        Verify if a property holds in the state space.
        
        Args:
            property_expr: Z3 boolean expression representing the property
            
        Returns:
            True if the property holds, False otherwise
        """
        # Save current state
        self.solver.push()
        
        # Add negation of property
        self.solver.add(Not(property_expr))
        
        # Check if negation is satisfiable
        result = self.solver.check() == unsat
        
        # Restore state
        self.solver.pop()
        
        return result 