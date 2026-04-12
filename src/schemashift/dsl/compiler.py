"""DSL compiler: Lark parse tree → polars.Expr.

:class:`DSLTransformer` walks the Lark CST bottom-up.  Scalar literals
(numbers, strings, booleans, null) are kept as Python values until they
are used in an expression context, where :meth:`_to_expr` wraps them in
``pl.lit()``.  This lets method arguments that require raw Python values
(e.g. ``round(n)``, ``cast("float64")``) be used directly without having
to unwrap a ``polars.Expr``.
"""

from __future__ import annotations

import ast as stdlib_ast
from typing import TYPE_CHECKING, Any, cast

import polars as pl
from lark import Token, Transformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars.type_aliases import TimeUnit

from schemashift.dtypes import DTYPE_MAP, DType
from schemashift.errors import DSLRuntimeError, DSLSyntaxError

from ._lookups import TABLES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ScalarValue = int | float | str | bool | None
_ExprOrScalar = pl.Expr | _ScalarValue


def _to_expr(v: _ExprOrScalar) -> pl.Expr:
    """Wrap a Python scalar in ``pl.lit()`` if it isn't already a ``polars.Expr``."""
    if isinstance(v, pl.Expr):
        return v
    return pl.lit(v)


def _require_str(v: _ExprOrScalar, method: str) -> str:
    if not isinstance(v, str):
        raise DSLSyntaxError(
            f"'{method}' requires a string literal argument, got {type(v).__name__}",
            expression="",
            position=-1,
        )
    return v


def _require_int(v: _ExprOrScalar, method: str) -> int:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise DSLSyntaxError(
            f"'{method}' requires a numeric literal argument, got {type(v).__name__}",
            expression="",
            position=-1,
        )
    return int(v)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


class DSLTransformer(Transformer):
    """Bottom-up Lark transformer that compiles a DSL parse tree to a ``polars.Expr``."""

    # ------------------------------------------------------------------
    # Scalars → Python values (wrapped by _to_expr only when needed)
    # ------------------------------------------------------------------

    def number(self, tok: list[Token]) -> _ScalarValue:
        return stdlib_ast.literal_eval(str(tok[0]))

    def string(self, tok: list[Token]) -> _ScalarValue:
        return stdlib_ast.literal_eval(str(tok[0]))

    def true_lit(self, _: list) -> bool:
        return True

    def false_lit(self, _: list) -> bool:
        return False

    def null_lit(self, _: list) -> None:
        return None

    # Map scalars — same logic, separate names to avoid confusion
    def map_number(self, tok: list[Token]) -> _ScalarValue:
        return stdlib_ast.literal_eval(str(tok[0]))

    def map_string(self, tok: list[Token]) -> _ScalarValue:
        return stdlib_ast.literal_eval(str(tok[0]))

    def map_true(self, _: list) -> bool:
        return True

    def map_false(self, _: list) -> bool:
        return False

    def map_null(self, _: list) -> None:
        return None

    # ------------------------------------------------------------------
    # Column reference
    # ------------------------------------------------------------------

    def col_ref(self, tok: list[Token]) -> pl.Expr:
        name = stdlib_ast.literal_eval(str(tok[0]))
        return pl.col(name)

    # ------------------------------------------------------------------
    # Root — ensures the top-level result is always a polars.Expr
    # ------------------------------------------------------------------

    def start(self, args: list[Any]) -> pl.Expr:
        return _to_expr(args[0])

    # ------------------------------------------------------------------
    # Postfix (method chaining entry point)
    # ------------------------------------------------------------------

    def postfix(self, args: list[Any]) -> Any:
        """Apply method-step callables to the atom left-to-right.

        When there are no method steps the atom value is returned as-is
        (Python scalar or polars.Expr); _to_expr wrapping happens at the
        operator level or at root via ``start()``.  When method steps are
        present the atom must first be converted to a polars.Expr.
        """
        if len(args) == 1:
            # No method steps — pass through raw (scalar or Expr)
            return args[0]
        obj = _to_expr(args[0])
        for apply_fn in args[1:]:
            obj = apply_fn(obj)
        return obj

    # ------------------------------------------------------------------
    # Unary operators
    # ------------------------------------------------------------------

    def unary_neg(self, args: list[Any]) -> pl.Expr:
        # args = [Token("MINUS", "-"), operand_expr] — MINUS is a named terminal so it's kept
        return pl.lit(0) - _to_expr(args[-1])

    def unary_not(self, args: list[Any]) -> pl.Expr:
        # "not" is an anonymous string literal, filtered from args → args = [operand_expr]
        return _to_expr(args[0]).not_()

    # ------------------------------------------------------------------
    # Binary operators (LALR keeps named terminals in args)
    # ------------------------------------------------------------------

    def logical(self, args: list[Any]) -> pl.Expr:
        result = _to_expr(args[0])
        for op, right in zip(args[1::2], args[2::2], strict=False):
            r = _to_expr(right)
            result = result & r if str(op) == "&" else result | r
        return result

    def comparison(self, args: list[Any]) -> pl.Expr:
        left = _to_expr(args[0])
        if len(args) == 1:
            return left
        op = str(args[1])
        right = _to_expr(args[2])
        dispatch: dict[str, pl.Expr] = {
            "==": left == right,
            "!=": left != right,
            ">": left > right,
            "<": left < right,
            ">=": left >= right,
            "<=": left <= right,
        }
        return dispatch[op]

    def additive(self, args: list[Any]) -> pl.Expr:
        result = _to_expr(args[0])
        for op, right in zip(args[1::2], args[2::2], strict=False):
            r = _to_expr(right)
            result = result + r if str(op) == "+" else result - r
        return result

    def multiplicative(self, args: list[Any]) -> pl.Expr:
        result = _to_expr(args[0])
        for op, right in zip(args[1::2], args[2::2], strict=False):
            r = _to_expr(right)
            op_str = str(op)
            if op_str == "*":
                result = result * r
            elif op_str == "/":
                result = result / r
            else:
                result = result % r
        return result

    # ------------------------------------------------------------------
    # Method steps — return callables (obj: pl.Expr) -> pl.Expr
    # ------------------------------------------------------------------

    def args(self, items: list[Any]) -> list[Any]:
        return list(items)

    def direct_method(self, items: list[Any]) -> Callable[[pl.Expr], pl.Expr]:  # noqa: C901
        method_name = str(items[0])
        method_args: list[Any] = items[1] if len(items) > 1 else []

        def apply(obj: pl.Expr) -> pl.Expr:  # noqa: C901
            match method_name:
                case "abs":
                    return obj.abs()
                case "is_null":
                    return obj.is_null()
                case "is_not_null":
                    return obj.is_not_null()
                case "is_nan":
                    return obj.is_nan()
                case "floor":
                    return obj.floor()
                case "ceil":
                    return obj.ceil()
                case "sqrt":
                    return obj.sqrt()
                case "round":
                    n = _require_int(method_args[0], "round")
                    return obj.round(n)
                case "cast":
                    type_str = _require_str(method_args[0], "cast").lower()
                    if type_str not in DTYPE_MAP:
                        raise DSLSyntaxError(
                            f"Unknown cast type: {type_str!r}. Valid types: {sorted(DTYPE_MAP)!r}",
                            expression="",
                            position=-1,
                        )
                    return obj.cast(DTYPE_MAP[cast("DType", type_str)])
                case "fill_null":
                    return obj.fill_null(_to_expr(method_args[0]))
                case "fill_nan":
                    return obj.fill_nan(_to_expr(method_args[0]))
                case "clip":
                    return obj.clip(method_args[0], method_args[1])
                case "pow":
                    return obj.pow(method_args[0])
                case _:
                    raise DSLRuntimeError(f"Unsupported direct method: {method_name!r}")

        return apply

    def str_method(self, items: list[Any]) -> Callable[[pl.Expr], pl.Expr]:  # noqa: C901
        method_name = str(items[0])
        method_args: list[Any] = items[1] if len(items) > 1 else []

        def apply(obj: pl.Expr) -> pl.Expr:  # noqa: C901
            match method_name:
                case "strip":
                    return obj.str.strip_chars()
                case "lstrip":
                    return obj.str.strip_chars_start()
                case "rstrip":
                    return obj.str.strip_chars_end()
                case "lower" | "to_lowercase":
                    return obj.str.to_lowercase()
                case "upper" | "to_uppercase":
                    return obj.str.to_uppercase()
                case "lengths":
                    return obj.str.len_chars()
                case "starts_with":
                    return obj.str.starts_with(_require_str(method_args[0], "str.starts_with"))
                case "ends_with":
                    return obj.str.ends_with(_require_str(method_args[0], "str.ends_with"))
                case "contains":
                    return obj.str.contains(_require_str(method_args[0], "str.contains"), literal=True)
                case "replace":
                    return obj.str.replace(
                        _require_str(method_args[0], "str.replace"),
                        _require_str(method_args[1], "str.replace"),
                        literal=True,
                    )
                case "replace_all":
                    return obj.str.replace_all(
                        _require_str(method_args[0], "str.replace_all"),
                        _require_str(method_args[1], "str.replace_all"),
                        literal=True,
                    )
                case "replace_regex":
                    return obj.str.replace(
                        _require_str(method_args[0], "str.replace_regex"),
                        _require_str(method_args[1], "str.replace_regex"),
                        literal=False,
                    )
                case "slice":
                    return obj.str.slice(
                        _require_int(method_args[0], "str.slice"),
                        _require_int(method_args[1], "str.slice"),
                    )
                case "to_datetime":
                    return obj.str.to_datetime(_require_str(method_args[0], "str.to_datetime"))
                case "extract":
                    return obj.str.extract(
                        _require_str(method_args[0], "str.extract"),
                        _require_int(method_args[1], "str.extract"),
                    )
                case "count_matches":
                    return obj.str.count_matches(_require_str(method_args[0], "str.count_matches"))
                case "zfill":
                    return obj.str.zfill(_require_int(method_args[0], "str.zfill"))
                case _:
                    raise DSLRuntimeError(f"Unsupported str method: {method_name!r}")

        return apply

    def dt_method(self, items: list[Any]) -> Callable[[pl.Expr], pl.Expr]:  # noqa: C901
        method_name = str(items[0])
        method_args: list[Any] = items[1] if len(items) > 1 else []

        def apply(obj: pl.Expr) -> pl.Expr:  # noqa: C901
            match method_name:
                case "year":
                    return obj.dt.year()
                case "month":
                    return obj.dt.month()
                case "day":
                    return obj.dt.day()
                case "quarter":
                    return obj.dt.quarter()
                case "week":
                    return obj.dt.week()
                case "weekday":
                    return obj.dt.weekday()
                case "ordinal_day":
                    return obj.dt.ordinal_day()
                case "hour":
                    return obj.dt.hour()
                case "minute":
                    return obj.dt.minute()
                case "second":
                    return obj.dt.second()
                case "millisecond":
                    return obj.dt.millisecond()
                case "microsecond":
                    return obj.dt.microsecond()
                case "strftime":
                    return obj.dt.strftime(_require_str(method_args[0], "dt.strftime"))
                case "timestamp":
                    unit = _require_str(method_args[0], "dt.timestamp") if method_args else "ms"
                    if unit not in ("ms", "us", "ns"):
                        raise DSLSyntaxError(
                            f"dt.timestamp() unit must be 'ms', 'us', or 'ns', got {unit!r}",
                            expression="",
                            position=-1,
                        )
                    return obj.dt.timestamp(cast("TimeUnit", unit))
                case "truncate":
                    return obj.dt.truncate(_require_str(method_args[0], "dt.truncate"))
                case "total_seconds":
                    return obj.dt.total_seconds()
                case _:
                    raise DSLRuntimeError(f"Unsupported dt method: {method_name!r}")

        return apply

    # ------------------------------------------------------------------
    # when/otherwise chain
    # ------------------------------------------------------------------

    def when_otherwise(self, args: list[Any]) -> tuple[list, Any]:
        """Terminal when_tail: (.otherwise(expr)) → ([], otherwise_expr)"""
        return ([], args[0])

    def when_more(self, args: list[Any]) -> tuple[list, Any]:
        """Recursive when_tail: (.when(cond, val) tail) → prepend to tail"""
        cond, val, (more_whens, otherwise) = args[0], args[1], args[2]
        return ([(cond, val), *more_whens], otherwise)

    def when_chain(self, args: list[Any]) -> pl.Expr:
        cond, val, (more_whens, otherwise) = args[0], args[1], args[2]
        all_whens = [(cond, val), *more_whens]
        first_cond, first_val = all_whens[0]
        chain = pl.when(_to_expr(first_cond)).then(_to_expr(first_val))
        for c, v in all_whens[1:]:
            chain = chain.when(_to_expr(c)).then(_to_expr(v))
        return chain.otherwise(_to_expr(otherwise))

    # ------------------------------------------------------------------
    # coalesce / lookup / custom_lookup
    # ------------------------------------------------------------------

    def coalesce_expr(self, args: list[Any]) -> pl.Expr:
        return pl.coalesce([_to_expr(a) for a in args])

    def lookup_expr(self, args: list[Any]) -> pl.Expr:
        expr = _to_expr(args[0])
        table_name = stdlib_ast.literal_eval(str(args[1]))
        if table_name not in TABLES:
            raise DSLSyntaxError(
                f"Unknown lookup table: {table_name!r}. Available: {sorted(TABLES)!r}",
                expression="",
                position=-1,
            )
        table = TABLES[table_name]
        return expr.replace(list(table.keys()), list(table.values()))

    def custom_lookup_expr(self, args: list[Any]) -> pl.Expr:
        expr = _to_expr(args[0])
        mapping_pairs: list[tuple] = args[1]
        base_name: str | None = stdlib_ast.literal_eval(str(args[2])) if len(args) > 2 else None
        if base_name is not None and base_name not in TABLES:
            raise DSLSyntaxError(
                f"Unknown base table: {base_name!r}. Available: {sorted(TABLES)!r}",
                expression="",
                position=-1,
            )
        base: dict = dict(TABLES[base_name]) if base_name else {}
        combined = {**base, **dict(mapping_pairs)}
        return expr.replace(list(combined.keys()), list(combined.values()))

    def map_lit(self, args: list[Any]) -> list[tuple]:
        return list(args)

    def map_entry(self, args: list[Any]) -> tuple[Any, Any]:
        return (args[0], args[1])
