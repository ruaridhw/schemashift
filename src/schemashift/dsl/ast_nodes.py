"""AST node definitions for the schemashift DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Literal:
    """A literal value: int, float, str, bool, or None."""

    value: Any


@dataclass(frozen=True)
class ColRef:
    """A reference to a DataFrame column by name."""

    name: str


@dataclass(frozen=True)
class BinaryOp:
    """A binary operator expression."""

    op: str  # +, -, *, /, %, ==, !=, >, <, >=, <=, &, |
    left: ASTNode
    right: ASTNode


@dataclass(frozen=True)
class UnaryOp:
    """A unary operator expression."""

    op: str  # - (negation), ! (not)
    operand: ASTNode


@dataclass(frozen=True)
class MethodCall:
    """A method call on an expression object.

    The ``method`` field uses dot-prefixed namespacing for sub-namespaces:
    e.g. ``"str.lower"``, ``"dt.year"``.  Top-level methods use plain
    names such as ``"round"``, ``"abs"``.
    """

    obj: ASTNode
    method: str
    args: tuple[ASTNode, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class WhenClause:
    """A single when/then pair inside a when-chain."""

    condition: ASTNode
    value: ASTNode


@dataclass(frozen=True)
class WhenChain:
    """A complete when/otherwise conditional expression."""

    whens: tuple[WhenClause, ...]
    otherwise: ASTNode


# Type alias covering every node variant.
ASTNode = Literal | ColRef | BinaryOp | UnaryOp | MethodCall | WhenClause | WhenChain
