"""
This module contains the LPSharedData class, which is used to store the shared data
between the LP nodes in an LP model.
"""

__docformat__ = "restructuredtext"
__all__ = ["LPSharedData"]

from collections import OrderedDict
from typing import Dict, List

import gurobipy


class LPSharedData:
    """
    This class is used to store the shared data between the LP nodes in an LP model.
    """

    def __init__(self):
        self.all_vars: Dict[str, List[gurobipy.Var]] = OrderedDict()
        """
        A dictionary that stores the gurobi variables of the LP nodes. The key is the
        name of the LP node, and the value is a list of gurobi variables.
        """

        self.all_constrs: Dict[str, List[gurobipy.Constr]] = OrderedDict()
        """
        A dictionary that stores the gurobi constraints of the LP nodes. The key is the
        name of the LP node, and the value is a list of gurobi constraints.
        """

        self.all_lower_bounds: Dict[str, List[float]] = OrderedDict()
        """
        A dictionary that stores the lower scalar bounds of the LP nodes. The key is the
        name of the LP node, and the value is a list of lower scalar bounds.
        """

        self.all_upper_bounds: Dict[str, List[float]] = OrderedDict()
        """
        A dictionary that stores the upper scalar bounds of the LP nodes. The key is the
        name of the LP node, and the value is a list of upper scalar bounds.
        """

        self.all_lower_relaxations: Dict[str, List] = OrderedDict()
        """
        A dictionary that stores the lower relaxations of the LP nodes. The key is the
        name of the LP node, and the value is a list of lower relaxations.
        """

        self.all_upper_relaxations: Dict[str, List] = OrderedDict()
        """
        A dictionary that stores the upper relaxations of the LP nodes. The key is the
        name of the LP node, and the value is a list of upper relaxations.
        """

    def clear(self):
        """
        Clear all the shared data.
        """
        self.all_vars.clear()
        self.all_constrs.clear()
        self.all_lower_bounds.clear()
        self.all_upper_bounds.clear()
        self.all_lower_relaxations.clear()
        self.all_upper_relaxations.clear()
