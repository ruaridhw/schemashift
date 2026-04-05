"""Compiler from DSL AST nodes to Polars expressions.

Takes the AST produced by :mod:`schemashift.dsl.parser` and converts it into
a ``polars.Expr`` that can be used directly inside ``df.select()``,
``df.with_columns()``, etc.
"""

from typing import cast

import polars as pl

from schemashift.dtypes import DTYPE_MAP, DType
from schemashift.errors import DSLRuntimeError, DSLSyntaxError

from ._lookups import TABLES
from .ast_nodes import (
    ASTNode,
    BinaryOp,
    Coalesce,
    ColRef,
    CustomLookup,
    Literal,
    Lookup,
    MethodCall,
    UnaryOp,
    WhenChain,
    WhenClause,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _literal_str(node: ASTNode, method: str) -> str:
    """Extract a Python string from a Literal node, for method args that need raw strings."""
    if not isinstance(node, Literal) or not isinstance(node.value, str):
        raise DSLSyntaxError(
            f"'{method}' requires a string literal argument",
            expression="",
            position=-1,
        )
    return node.value


def _literal_int(node: ASTNode, method: str) -> int:
    """Extract a Python int from a Literal node."""
    if not isinstance(node, Literal) or not isinstance(node.value, int) or isinstance(node.value, bool):
        raise DSLSyntaxError(
            f"'{method}' requires an integer literal argument",
            expression="",
            position=-1,
        )
    return node.value


def _literal_int_or_float(node: ASTNode, method: str) -> int | float:
    """Extract a Python int or float from a Literal node (for round(n))."""
    if not isinstance(node, Literal) or not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
        raise DSLSyntaxError(
            f"'{method}' requires a numeric literal argument",
            expression="",
            position=-1,
        )
    return node.value


# ---------------------------------------------------------------------------
# Core compile function
# ---------------------------------------------------------------------------


def compile_dsl(node: ASTNode) -> pl.Expr:  # noqa: C901 (complex but intentional dispatch)
    """Compile *node* to a :class:`polars.Expr`.

    Raises :class:`schemashift.errors.DSLSyntaxError` for invalid node structure
    and :class:`schemashift.errors.DSLRuntimeError` for unsupported operations.
    """
    match node:
        case Literal(value):
            return pl.lit(value)

        case ColRef(name):
            return pl.col(name)

        case UnaryOp(op="-", operand=operand):
            return pl.lit(0) - compile_dsl(operand)

        case UnaryOp(op="!", operand=operand):
            return compile_dsl(operand).not_()

        case UnaryOp(op=op):
            raise DSLRuntimeError(f"Unsupported unary operator: {op!r}")

        case BinaryOp(op, left, right):
            l_expr = compile_dsl(left)
            r_expr = compile_dsl(right)
            match op:
                case "+":
                    return l_expr + r_expr
                case "-":
                    return l_expr - r_expr
                case "*":
                    return l_expr * r_expr
                case "/":
                    return l_expr / r_expr
                case "%":
                    return l_expr % r_expr
                case "==":
                    return l_expr == r_expr
                case "!=":
                    return l_expr != r_expr
                case ">":
                    return l_expr > r_expr
                case "<":
                    return l_expr < r_expr
                case ">=":
                    return l_expr >= r_expr
                case "<=":
                    return l_expr <= r_expr
                case "&":
                    return l_expr & r_expr
                case "|":
                    return l_expr | r_expr
                case _:
                    raise DSLRuntimeError(f"Unsupported binary operator: {op!r}")

        case MethodCall(obj, method, args):
            return _compile_method(obj, method, args)

        case WhenChain(whens, otherwise):
            return _compile_when_chain(whens, otherwise)

        case WhenClause():
            raise DSLRuntimeError("WhenClause cannot be compiled standalone; use WhenChain.")

        case Coalesce(exprs):
            return pl.coalesce([compile_dsl(e) for e in exprs])

        case Lookup(expr, table_name):

            if table_name not in TABLES:
                raise DSLSyntaxError(
                    f"Unknown lookup table: {table_name!r}. Available: {sorted(TABLES)!r}",
                    expression="",
                    position=-1,
                )
            table = TABLES[table_name]
            return compile_dsl(expr).replace(list(table.keys()), list(table.values()))

        case CustomLookup(expr, mapping, base_table):

            if base_table is not None:
                if base_table not in TABLES:
                    raise DSLSyntaxError(
                        f"Unknown base table: {base_table!r}. Available: {sorted(TABLES)!r}",
                        expression="",
                        position=-1,
                    )
                combined: dict = dict(TABLES[base_table])
            else:
                combined = {}
            for k, v in mapping:
                combined[k.value] = v.value
            return compile_dsl(expr).replace(list(combined.keys()), list(combined.values()))

        case _:
            raise DSLRuntimeError(f"Unknown AST node type: {type(node).__name__}")


def _compile_method(obj: ASTNode, method: str, args: tuple[ASTNode, ...]) -> pl.Expr:  # noqa: C901
    """Dispatch a MethodCall to the appropriate Polars expression."""
    base = compile_dsl(obj)

    match method:
        # ------------------------------------------------------------------
        # Direct methods
        # ------------------------------------------------------------------
        case "abs":
            _expect_arity(method, args, 0)
            return base.abs()

        case "is_null":
            _expect_arity(method, args, 0)
            return base.is_null()

        case "round":
            _expect_arity(method, args, 1)
            n = _literal_int_or_float(args[0], method)
            return base.round(int(n))

        case "cast":
            _expect_arity(method, args, 1)
            type_str = _literal_str(args[0], method).lower()
            if type_str not in DTYPE_MAP:
                raise DSLSyntaxError(
                    f"Unknown cast type: {type_str!r}. Valid types: {sorted(DTYPE_MAP)!r}",
                    expression="",
                    position=-1,
                )
            return base.cast(DTYPE_MAP[cast("DType", type_str)])

        case "fill_null":
            _expect_arity(method, args, 1)
            return base.fill_null(compile_dsl(args[0]))

        # ------------------------------------------------------------------
        # String methods
        # ------------------------------------------------------------------
        case "str.strip":
            _expect_arity(method, args, 0)
            return base.str.strip_chars()

        case "str.lower" | "str.to_lowercase":
            _expect_arity(method, args, 0)
            return base.str.to_lowercase()

        case "str.upper" | "str.to_uppercase":
            _expect_arity(method, args, 0)
            return base.str.to_uppercase()

        case "str.lengths":
            _expect_arity(method, args, 0)
            return base.str.len_chars()

        case "str.slice":
            _expect_arity(method, args, 2)
            offset = _literal_int(args[0], method)
            length = _literal_int(args[1], method)
            return base.str.slice(offset, length)

        case "str.replace":
            _expect_arity(method, args, 2)
            old = _literal_str(args[0], method)
            new = _literal_str(args[1], method)
            return base.str.replace(old, new, literal=True)

        case "str.replace_regex":
            _expect_arity(method, args, 2)
            pattern = _literal_str(args[0], method)
            replacement = _literal_str(args[1], method)
            return base.str.replace(pattern, replacement, literal=False)

        case "str.contains":
            _expect_arity(method, args, 1)
            pat = _literal_str(args[0], method)
            return base.str.contains(pat, literal=True)

        case "str.starts_with":
            _expect_arity(method, args, 1)
            prefix = _literal_str(args[0], method)
            return base.str.starts_with(prefix)

        case "str.ends_with":
            _expect_arity(method, args, 1)
            suffix = _literal_str(args[0], method)
            return base.str.ends_with(suffix)

        case "str.to_datetime":
            _expect_arity(method, args, 1)
            fmt = _literal_str(args[0], method)
            return base.str.to_datetime(fmt)

        case "str.extract":
            _expect_arity(method, args, 2)
            pattern = _literal_str(args[0], method)
            group_index = _literal_int(args[1], method)
            return base.str.extract(pattern, group_index)

        # ------------------------------------------------------------------
        # Datetime methods
        # ------------------------------------------------------------------
        case "dt.year":
            _expect_arity(method, args, 0)
            return base.dt.year()

        case "dt.month":
            _expect_arity(method, args, 0)
            return base.dt.month()

        case "dt.day":
            _expect_arity(method, args, 0)
            return base.dt.day()

        case "dt.hour":
            _expect_arity(method, args, 0)
            return base.dt.hour()

        case "dt.minute":
            _expect_arity(method, args, 0)
            return base.dt.minute()

        case "dt.second":
            _expect_arity(method, args, 0)
            return base.dt.second()

        case "dt.strftime":
            _expect_arity(method, args, 1)
            fmt = _literal_str(args[0], method)
            return base.dt.strftime(fmt)

        case "dt.timestamp":
            if len(args) == 0:
                return base.dt.timestamp("ms")
            _expect_arity(method, args, 1)
            unit = _literal_str(args[0], method)
            if unit not in ("ms", "us", "ns"):
                raise DSLSyntaxError(
                    f"dt.timestamp() unit must be 'ms', 'us', or 'ns', got {unit!r}",
                    expression="",
                    position=-1,
                )
            return base.dt.timestamp(unit)  # ty: ignore[invalid-argument-type]

        case _:
            raise DSLRuntimeError(f"Unsupported method: {method!r}")


def _expect_arity(method: str, args: tuple[ASTNode, ...], expected: int) -> None:
    if len(args) != expected:
        raise DSLSyntaxError(
            f"'{method}' expects {expected} argument(s), got {len(args)}",
            expression="",
            position=-1,
        )


def _compile_when_chain(whens: tuple[WhenClause, ...], otherwise: ASTNode) -> pl.Expr:
    """Build a chained pl.when(...).then(...).when(...).then(...).otherwise(...)."""
    if not whens:
        raise DSLRuntimeError("WhenChain must have at least one WhenClause.")

    first = whens[0]
    chain: pl.Expr = pl.when(compile_dsl(first.condition)).then(compile_dsl(first.value))
    for clause in whens[1:]:
        chain = chain.when(compile_dsl(clause.condition)).then(compile_dsl(clause.value))
    return chain.otherwise(compile_dsl(otherwise))
