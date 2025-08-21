"""
This module contains the type hints used in this package and these types can be used in
type checking.
"""

__docformat__ = "restructuredtext"
__all__ = ["NPFloatType", "CDDNumberType"]

from typing import Literal

import numpy as np

# Numpy dtype.
NPFloatType = (
    np.float64.__class__
    | np.float32.__class__
    | type(np.dtype("float64"))
    | type(np.dtype("float32"))
)
"""The numpy float type."""

CDDNumberType = Literal["float", "fraction"]
"""The number type used in the cdd library."""
