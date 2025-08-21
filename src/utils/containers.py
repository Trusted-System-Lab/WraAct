"""
This module contains the data structures used in this package.
These data structures are commonly used to store the intermediate results and improve
the readability of the code.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "ScalarBound",
    "LConstr",
    "LConstrBound",
]

from dataclasses import dataclass

import torch
from torch import Tensor

_TOLERANCE = 1e-8


@dataclass
class ScalarBound:
    """
    A class of two tensors representing lower and upper scalar bounds.

    :exception ValueError: If the shapes of l and u are not the same.
    """

    l: Tensor
    """The scalar lower bounds."""

    u: Tensor = None
    """The scalar upper bounds."""

    def __post_init__(self):
        if self.u is not None:
            if self.l.shape != self.u.shape:
                raise ValueError("The shapes of l and u must be the same.")
            if torch.any(self.l > self.u + _TOLERANCE):
                mask = self.l > self.u
                raise ValueError(
                    "The lower bound must be less than the upper bound. "
                    f"but \n{self.l[mask].tolist()}\n{self.u[mask].tolist()}."
                )

    def to(self, *args, **kwargs) -> "ScalarBound":
        """This is a similar function to the `torch.Tensor.to()` function."""

        self.l.to(*args, **kwargs)
        if self.u is not None:
            self.u.to(*args, **kwargs)
        return self

    def clone(self) -> "ScalarBound":
        """This is a similar function to the `torch.Tensor.clone()` function."""

        new_bound = ScalarBound(l=self.l.clone())
        if self.u is not None:
            new_bound.u = self.u.clone()
        return new_bound

    def detach(self) -> "ScalarBound":
        """This is a similar function to the `torch.Tensor.detach()` function."""

        new_bound = ScalarBound(l=self.l.detach())
        if self.u is not None:
            new_bound.u = self.u.detach()
        return new_bound

    def detach_(self) -> "ScalarBound":
        """This is a similar function to the `torch.Tensor.detach_()` function."""

        self.l.detach_()
        if self.u is not None:
            self.u.detach_()
        return self

    def requires_grad(self) -> bool:
        """This is a similar function to the `torch.Tensor.requires_grad()` function."""
        return self.l.requires_grad or (self.u is not None and self.u.requires_grad)

    def requires_grad_(self, requires_grad: bool) -> "ScalarBound":
        """
        This is a similar function to the `torch.Tensor.requires_grad_()` function.
        """

        self.l.requires_grad_(requires_grad)
        if self.u is not None:
            self.u.requires_grad_(requires_grad)
        return self

    def intersect(self, other: "ScalarBound") -> "ScalarBound":
        """This function computes the intersection of two scalar bounds as intervals."""
        # logger = logging.getLogger("rover")

        l = torch.maximum(self.l, other.l)
        # logger.debug(
        #     f"Lower bounds increase avg "
        #     f"{l.mean():.2f} - {self.l.mean():.2f} = {(l - self.l).mean():.2f}."
        # )
        u = None
        if self.u is not None and other.u is not None:
            u = torch.minimum(self.u, other.u)
            # logger.debug(
            #     f"Upper bounds decrease avg "
            #     f"{u.mean():.2f} - {self.u.mean():.2f} = {(u - self.u).mean():.2f}."
            # )
        elif self.u is not None:
            u = self.u
        elif other.u is not None:
            u = other.u

        return ScalarBound(l=l, u=u)

    def reshape(self, *shape: int) -> "ScalarBound":
        """This function reshapes the scalar bounds."""
        return ScalarBound(
            l=self.l.reshape(*shape),
            u=None if self.u is None else self.u.reshape(*shape),
        )

    def __str__(self):
        return (
            f"ScalarBound("
            f"l={self.l.tolist()}, "
            f"u={self.u.tolist() if self.u is not None else None}"
            f")"
        )

    def __repr__(self):
        return self.__str__()


@dataclass
class LConstr:
    """
    A class of two tensors representing linear constraints.

    :exception ValueError: If the first dimension of A and b are not the same.

    .. tip::

        The constraints are represented in the form of :math:`b + Ax >= 0`.
        The shape of :math:`A` is (num_constraints, n) or (num_constraints, n,
        n) and the shape of :math:`b` is (num_constraints,), where (num_constraints,
        n) for linear layer and (num_constraints, n, n) for convolutional layer.
    """

    A: Tensor
    """The coefficients of variables in the constraints."""

    b: Tensor | None = None
    """The constant items of the constraints."""

    def __post_init__(self):
        if self.b is not None:
            if self.A.shape[0] != self.b.shape[0]:
                raise ValueError("The first dimension of A and b must be the same.")

    def to(self, *args, **kwargs) -> "LConstr":
        """This is a similar function to the `torch.Tensor.to()` function."""

        self.A.to(*args, **kwargs)
        if self.b is not None:
            self.b.to(*args, **kwargs)
        return self

    def clone(self) -> "LConstr":
        """This is a similar function to the `torch.Tensor.clone()` function."""

        return LConstr(A=self.A.clone(), b=None if self.b is None else self.b.clone())

    def detach(self) -> "LConstr":
        """This is a similar function to the `torch.Tensor.detach()` function."""

        return LConstr(A=self.A.detach(), b=None if self.b is None else self.b.detach())

    def detach_(self) -> "LConstr":
        """This is a similar function to the `torch.Tensor.detach_()` function."""

        self.A.detach_()
        if self.b is not None:
            self.b.detach_()
        return self

    def requires_grad(self) -> bool:
        """This is a similar function to the `torch.Tensor.requires_grad()` function."""
        return self.A.requires_grad or (self.b is not None and self.b.requires_grad)

    def requires_grad_(self, requires_grad: bool) -> "LConstr":
        """
        This is a similar function to the `torch.Tensor.requires_grad_()` function.
        """
        self.A.requires_grad_(requires_grad)
        if self.b is not None:
            self.b.requires_grad_(requires_grad)
        return self

    def __add__(self, other: "LConstr") -> "LConstr":
        A = self.A + other.A
        b = None
        if self.b is not None and other.b is not None:
            b = self.b + other.b
        elif self.b is not None:
            b = self.b
        elif other.b is not None:
            b = other.b
        return LConstr(A=A, b=b)

    def __sub__(self, other: "LConstr") -> "LConstr":
        A = self.A - other.A
        b = None
        if self.b is not None and other.b is not None:
            b = self.b - other.b
        elif self.b is not None:
            b = self.b
        elif other.b is not None:
            b = other.b
        return LConstr(A=A, b=b)

    def __str__(self):
        return (
            f"LConstr(A={self.A.tolist()}, b"
            f"={self.b.tolist() if self.b is not None else None})"
        )

    def __repr__(self):
        return self.__str__()


@dataclass
class LConstrBound:
    """
    A class of two constraints representing lower and upper bounds. The lower constraint
    bound must not be none. But the upper constraint bound can be none, which serves
    to most verification for the last output variables.

    :exception ValueError: If the shapes of A are not the same.
    :exception ValueError: If the shapes of b are not the same.
    """

    L: LConstr
    """The lower constraint bound."""

    U: LConstr | None = None
    """The upper constraint bound."""

    def __post_init__(self):
        if self.U is not None:
            if self.L.A.shape != self.U.A.shape:
                raise ValueError("The shapes of A must be the same.")

            if self.L.b is not None and self.U.b is not None:
                if self.L.b.shape != self.U.b.shape:
                    raise ValueError("The shapes of b must be the same.")

    def to(self, *args, **kwargs) -> "LConstrBound":
        """This is a similar function to the `torch.Tensor.to()` function."""

        self.L.to(*args, **kwargs)
        if self.U is not None:
            self.U.to(*args, **kwargs)
        return self

    def clone(self) -> "LConstrBound":
        """This is a similar function to the `torch.Tensor.clone()` function."""

        new_constr_bound = LConstrBound(L=self.L.clone())
        if self.U is not None:
            new_constr_bound.U = self.U.clone()
        return new_constr_bound

    def detach(self) -> "LConstrBound":
        """This is a similar function to the `torch.Tensor.detach()` function."""

        new_constr_bound = LConstrBound(L=self.L.detach())
        if self.U is not None:
            new_constr_bound.U = self.U.detach()
        return new_constr_bound

    def detach_(self) -> "LConstrBound":
        """This is a similar function to the `torch.Tensor.detach_()` function."""

        self.L.detach_()
        if self.U is not None:
            self.U.detach_()
        return self

    def requires_grad(self) -> bool:
        """This is a similar function to the `torch.Tensor.requires_grad()` function."""
        return self.L.requires_grad or (self.U is not None and self.U.requires_grad)

    def requires_grad_(self, requires_grad: bool) -> "LConstrBound":
        """
        This is a similar function to the `torch.Tensor.requires_grad_()` function.
        """

        self.L.requires_grad_(requires_grad)
        if self.U is not None:
            self.U.requires_grad_(requires_grad)
        return self

    def __add__(self, other: "LConstrBound") -> "LConstrBound":
        L = self.L + other.L
        U = None
        if self.U is not None and other.U is not None:
            U = self.U + other.U
        elif self.U is not None:
            U = self.U
        elif other.U is not None:
            U = other.U
        return LConstrBound(L=L, U=U)

    def __sub__(self, other: "LConstrBound") -> "LConstrBound":
        L = self.L - other.L
        U = None
        if self.U is not None and other.U is not None:
            U = self.U - other.U
        elif self.U is not None:
            U = self.U
        elif other.U is not None:
            U = other.U
        return LConstrBound(L=L, U=U)
