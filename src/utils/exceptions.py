"""
This module contains the customized exceptions used in this package.
"""

__docformat__ = "restructuredtext"
__all__ = ["NotSupported", "NotConverged", "Degenerated"]


class NotSupported(Exception):
    """
    An exception for not supported functions or methods.

    :param message: The shown exception message.
    """

    def __init__(
        self,
        message="This is not supported, and it should not be called. It may be "
        "implemented implicitly in another way.",
    ):
        super().__init__(message)

    def __str__(self):
        return f"{self.__class__.__name__}: {super().__str__()}"


class NotConverged(Exception):
    """
    An exception for not converged calculation in an algorithm.

    :param message: The shown exception message.
    """

    def __init__(
        self,
        message="The calculation is not converged after the maximum number of "
        "iterations.",
    ):
        super().__init__(message)

    def __str__(self):
        return f"{self.__class__.__name__}: {super().__str__()}"


class Degenerated(Exception):
    """
    An exception for degenerated input polytope when calculating function hull.
    It means the number of vertices is fewer than the dimension
    """

    def __init__(
        self,
        message="The polytope is degenerated.",
    ):
        super().__init__(message)

    def __str__(self):
        return f"{self.__class__.__name__}: {super().__str__()}"
