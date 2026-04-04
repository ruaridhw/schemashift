"""Recursive-descent parser for the schemashift DSL.

Converts a DSL expression string into an AST composed of nodes from
:mod:`schemashift.dsl.ast_nodes`.  Raises :class:`schemashift.errors.DSLSyntaxError`
for any invalid input.
"""

import ast
import re
from enum import Enum, auto
from typing import NamedTuple

from schemashift.errors import DSLSyntaxError

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
# Allowlisted methods
# ---------------------------------------------------------------------------

_DIRECT_METHODS: frozenset[str] = frozenset({"round", "abs", "cast", "fill_null", "is_null"})

_STR_METHODS: frozenset[str] = frozenset(
    {
        "strip",
        "lower",
        "upper",
        "to_lowercase",
        "to_uppercase",
        "slice",
        "replace",
        "replace_regex",
        "contains",
        "starts_with",
        "ends_with",
        "to_datetime",
        "lengths",
        "extract",
    }
)

_DT_METHODS: frozenset[str] = frozenset({"year", "month", "day", "hour", "minute", "second", "strftime", "timestamp"})


def _is_allowed_method(namespace: str, name: str) -> bool:
    if namespace == "":
        return name in _DIRECT_METHODS
    if namespace == "str":
        return name in _STR_METHODS
    if namespace == "dt":
        return name in _DT_METHODS
    return False


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class TT(Enum):
    """Token type."""

    NUMBER = auto()
    STRING = auto()
    IDENT = auto()
    DOT = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    EQ = auto()  # ==
    NE = auto()  # !=
    GT = auto()  # >
    LT = auto()  # <
    GE = auto()  # >=
    LE = auto()  # <=
    AMP = auto()  # &
    PIPE = auto()  # |
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    COLON = auto()  # :
    EOF = auto()


class Token(NamedTuple):
    type: TT
    value: object  # str | int | float
    pos: int  # character offset in source


# Regex patterns for tokenizing (order matters).
_TOKEN_RE = re.compile(
    r"""
    (?P<FLOAT> \d+\.\d* | \.\d+)       # float before int
    |(?P<INT>   \d+)
    |(?P<STRING>  '[^'\\]*(?:\\.[^'\\]*)*' | "[^"\\]*(?:\\.[^"\\]*)*")
    |(?P<IDENT>   [A-Za-z_][A-Za-z0-9_]*)
    |(?P<EQ>    ==)
    |(?P<NE>    !=)
    |(?P<GE>    >=)
    |(?P<LE>    <=)
    |(?P<GT>    >)
    |(?P<LT>    <)
    |(?P<DOT>   \.)
    |(?P<LPAREN> \()
    |(?P<RPAREN> \))
    |(?P<COMMA>  ,)
    |(?P<PLUS>   \+)
    |(?P<MINUS>  -)
    |(?P<STAR>   \*)
    |(?P<SLASH>  /)
    |(?P<PERCENT> %)
    |(?P<AMP>    &)
    |(?P<PIPE>   \|)
    |(?P<LBRACE>  \{)
    |(?P<RBRACE>  \})
    |(?P<COLON>   :)
    |(?P<WS>    \s+)                    # ignored whitespace
    """,
    re.VERBOSE,
)

_TT_MAP: dict[str, TT] = {
    "EQ": TT.EQ,
    "NE": TT.NE,
    "GE": TT.GE,
    "LE": TT.LE,
    "GT": TT.GT,
    "LT": TT.LT,
    "DOT": TT.DOT,
    "LPAREN": TT.LPAREN,
    "RPAREN": TT.RPAREN,
    "COMMA": TT.COMMA,
    "PLUS": TT.PLUS,
    "MINUS": TT.MINUS,
    "STAR": TT.STAR,
    "SLASH": TT.SLASH,
    "PERCENT": TT.PERCENT,
    "AMP": TT.AMP,
    "PIPE": TT.PIPE,
    "LBRACE": TT.LBRACE,
    "RBRACE": TT.RBRACE,
    "COLON": TT.COLON,
    "IDENT": TT.IDENT,
}


def _unescape_string(raw: str) -> str:
    """Remove surrounding quotes and process escape sequences.

    Delegates to :func:`ast.literal_eval` so that ``\\n``, ``\\t``,
    ``\\uXXXX``, etc. are interpreted the same way Python would.
    """
    # ast.literal_eval handles both single- and double-quoted strings.
    return ast.literal_eval(raw)


def tokenize(expression: str) -> list[Token]:
    """Convert *expression* to a flat list of tokens (excluding whitespace)."""
    tokens: list[Token] = []
    pos = 0
    length = len(expression)
    while pos < length:
        m = _TOKEN_RE.match(expression, pos)
        if m is None:
            raise DSLSyntaxError(
                f"Unexpected character {expression[pos]!r}",
                expression=expression,
                position=pos,
            )
        kind = m.lastgroup
        text = m.group()
        if kind == "WS":
            pos = m.end()
            continue
        if kind == "FLOAT":
            tokens.append(Token(TT.NUMBER, float(text), pos))
        elif kind == "INT":
            tokens.append(Token(TT.NUMBER, int(text), pos))
        elif kind == "STRING":
            tokens.append(Token(TT.STRING, _unescape_string(text), pos))
        else:
            if kind is None:  # pragma: no cover
                raise DSLSyntaxError(f"Unexpected token at position {pos}", expression=expression, position=pos)
            tt = _TT_MAP[kind]
            tokens.append(Token(tt, text, pos))
        pos = m.end()
    tokens.append(Token(TT.EOF, "", length))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_KEYWORDS: frozenset[str] = frozenset(
    {"col", "when", "otherwise", "true", "false", "null", "coalesce", "lookup", "custom_lookup", "not"}
)


class _Parser:
    """Hand-written recursive-descent parser."""

    def __init__(self, expression: str) -> None:
        self._expr = expression
        self._tokens = tokenize(expression)
        self._pos = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, tt: TT, *, hint: str = "") -> Token:
        tok = self._peek()
        if tok.type != tt:
            msg = hint or f"Expected {tt.name}, got {tok.value!r}"
            raise DSLSyntaxError(msg, expression=self._expr, position=tok.pos)
        return self._advance()

    def _match(self, *types: TT) -> bool:
        return self._peek().type in types

    def _current_pos(self) -> int:
        return self._peek().pos

    # ------------------------------------------------------------------
    # Grammar productions
    # ------------------------------------------------------------------

    def parse(self) -> ASTNode:
        node = self._expression()
        eof = self._peek()
        if eof.type != TT.EOF:
            raise DSLSyntaxError(
                f"Unexpected token {eof.value!r} after expression",
                expression=self._expr,
                position=eof.pos,
            )
        return node

    # expression := logical
    def _expression(self) -> ASTNode:
        return self._logical()

    # logical := comparison (('&' | '|') comparison)*
    def _logical(self) -> ASTNode:
        left = self._comparison()
        while self._match(TT.AMP, TT.PIPE):
            op_tok = self._advance()
            op = "&" if op_tok.type == TT.AMP else "|"
            right = self._comparison()
            left = BinaryOp(op, left, right)
        return left

    # comparison := additive (('==' | '!=' | '>' | '<' | '>=' | '<=') additive)?
    def _comparison(self) -> ASTNode:
        left = self._additive()
        _CMP = {TT.EQ: "==", TT.NE: "!=", TT.GT: ">", TT.LT: "<", TT.GE: ">=", TT.LE: "<="}
        if self._peek().type in _CMP:
            op_tok = self._advance()
            op = _CMP[op_tok.type]
            right = self._additive()
            left = BinaryOp(op, left, right)
        return left

    # additive := multiplicative (('+' | '-') multiplicative)*
    def _additive(self) -> ASTNode:
        left = self._multiplicative()
        while self._match(TT.PLUS, TT.MINUS):
            op_tok = self._advance()
            op = "+" if op_tok.type == TT.PLUS else "-"
            right = self._multiplicative()
            left = BinaryOp(op, left, right)
        return left

    # multiplicative := unary (('*' | '/' | '%') unary)*
    def _multiplicative(self) -> ASTNode:
        left = self._unary()
        _MUL = {TT.STAR: "*", TT.SLASH: "/", TT.PERCENT: "%"}
        while self._peek().type in _MUL:
            op_tok = self._advance()
            op = _MUL[op_tok.type]
            right = self._unary()
            left = BinaryOp(op, left, right)
        return left

    # unary := '-' unary | 'not' unary | atom_with_methods
    def _unary(self) -> ASTNode:
        if self._match(TT.MINUS):
            self._advance()
            operand = self._unary()
            return UnaryOp("-", operand)
        if self._match(TT.IDENT) and str(self._peek().value) == "not":
            self._advance()
            operand = self._unary()
            return UnaryOp("!", operand)
        return self._atom_with_methods()

    # atom_with_methods := atom ('.' method_chain)*
    def _atom_with_methods(self) -> ASTNode:
        node = self._atom()
        while self._match(TT.DOT):
            node = self._method_chain_step(node)
        return node

    # method_chain := IDENT '(' args? ')' | IDENT '.' IDENT '(' args? ')'
    def _method_chain_step(self, obj: ASTNode) -> ASTNode:
        self._expect(TT.DOT)
        name_tok = self._expect(TT.IDENT, hint="Expected method name after '.'")
        name = str(name_tok.value)

        # Sub-namespace: str.xxx or dt.xxx
        if self._match(TT.DOT):
            # namespace.method
            self._advance()  # consume second dot
            sub_tok = self._expect(TT.IDENT, hint=f"Expected method name after '{name}.'")
            sub_name = str(sub_tok.value)
            full_method = f"{name}.{sub_name}"
            if not _is_allowed_method(name, sub_name):
                raise DSLSyntaxError(
                    f"Unknown method '{full_method}'",
                    expression=self._expr,
                    position=sub_tok.pos,
                )
            self._expect(TT.LPAREN, hint=f"Expected '(' after '{full_method}'")
            args = self._args() if not self._match(TT.RPAREN) else ()
            self._expect(TT.RPAREN, hint=f"Expected ')' to close '{full_method}(...'")
            return MethodCall(obj, full_method, tuple(args))
        # Direct method
        if not _is_allowed_method("", name):
            raise DSLSyntaxError(
                f"Unknown method '{name}'",
                expression=self._expr,
                position=name_tok.pos,
            )
        self._expect(TT.LPAREN, hint=f"Expected '(' after '{name}'")
        args = self._args() if not self._match(TT.RPAREN) else ()
        self._expect(TT.RPAREN, hint=f"Expected ')' to close '{name}(...'")
        return MethodCall(obj, name, tuple(args))

    # atom := NUMBER | STRING | BOOLEAN | NULL | col_ref | when_expr | '(' expression ')'
    def _atom(self) -> ASTNode:
        tok = self._peek()

        if tok.type == TT.NUMBER:
            self._advance()
            return Literal(tok.value)

        if tok.type == TT.STRING:
            self._advance()
            return Literal(tok.value)

        if tok.type == TT.IDENT:
            ident = str(tok.value)
            if ident == "true":
                self._advance()
                return Literal(True)
            if ident == "false":
                self._advance()
                return Literal(False)
            if ident == "null":
                self._advance()
                return Literal(None)
            if ident == "col":
                return self._col_ref()
            if ident == "when":
                return self._when_expr()
            if ident == "coalesce":
                return self._coalesce_expr()
            if ident == "lookup":
                return self._lookup_expr()
            if ident == "custom_lookup":
                return self._custom_lookup_expr()
            if ident == "not":
                raise DSLSyntaxError(
                    "'not' must prefix an expression (e.g. not col(\"x\").is_null())",
                    expression=self._expr,
                    position=tok.pos,
                )
            # Unknown identifier
            raise DSLSyntaxError(
                f"Unknown identifier {ident!r}",
                expression=self._expr,
                position=tok.pos,
            )

        if tok.type == TT.LPAREN:
            self._advance()
            node = self._expression()
            self._expect(TT.RPAREN, hint="Expected ')' to close parenthesised expression")
            return node

        raise DSLSyntaxError(
            f"Unexpected token {tok.value!r}",
            expression=self._expr,
            position=tok.pos,
        )

    # col_ref := 'col' '(' STRING ')'
    def _col_ref(self) -> ColRef:
        self._expect(TT.IDENT, hint="Expected 'col'")
        self._expect(TT.LPAREN, hint="Expected '(' after 'col'")
        name_tok = self._expect(TT.STRING, hint="Expected column name string in col(...)")
        self._expect(TT.RPAREN, hint="Expected ')' after column name")
        return ColRef(str(name_tok.value))

    # when_expr := 'when' '(' expr ',' expr ')' ('.when(...)*)* '.otherwise' '(' expr ')'
    def _when_expr(self) -> WhenChain:
        # Consume leading 'when'
        start_tok = self._expect(TT.IDENT, hint="Expected 'when'")
        if str(start_tok.value) != "when":
            raise DSLSyntaxError(
                "Expected 'when'",
                expression=self._expr,
                position=start_tok.pos,
            )
        whens: list[WhenClause] = []
        # First when(..., ...)
        self._expect(TT.LPAREN, hint="Expected '(' after 'when'")
        cond = self._expression()
        self._expect(TT.COMMA, hint="Expected ',' between when condition and value")
        val = self._expression()
        self._expect(TT.RPAREN, hint="Expected ')' to close 'when(...'")
        whens.append(WhenClause(cond, val))

        # Chain: .when(...) | .otherwise(...)
        while self._match(TT.DOT):
            self._advance()  # consume '.'
            kw_tok = self._expect(TT.IDENT, hint="Expected 'when' or 'otherwise' after '.'")
            kw = str(kw_tok.value)
            if kw == "when":
                self._expect(TT.LPAREN, hint="Expected '(' after '.when'")
                cond2 = self._expression()
                self._expect(TT.COMMA, hint="Expected ',' between when condition and value")
                val2 = self._expression()
                self._expect(TT.RPAREN, hint="Expected ')' to close '.when(...'")
                whens.append(WhenClause(cond2, val2))
            elif kw == "otherwise":
                self._expect(TT.LPAREN, hint="Expected '(' after '.otherwise'")
                default = self._expression()
                self._expect(TT.RPAREN, hint="Expected ')' to close '.otherwise(...'")
                return WhenChain(tuple(whens), default)
            else:
                raise DSLSyntaxError(
                    f"Expected 'when' or 'otherwise', got {kw!r}",
                    expression=self._expr,
                    position=kw_tok.pos,
                )
        # Fell off the end without .otherwise(...)
        raise DSLSyntaxError(
            "Expected '.otherwise(...)' to close when-chain",
            expression=self._expr,
            position=self._current_pos(),
        )

    # coalesce_expr := 'coalesce' '(' expression ',' expression (',' expression)* ')'
    def _coalesce_expr(self) -> Coalesce:
        self._expect(TT.IDENT, hint="Expected 'coalesce'")
        self._expect(TT.LPAREN, hint="Expected '(' after 'coalesce'")
        exprs = self._args()
        self._expect(TT.RPAREN, hint="Expected ')' to close 'coalesce(...'")
        if len(exprs) < 2:
            raise DSLSyntaxError(
                "coalesce() requires at least 2 arguments",
                expression=self._expr,
                position=self._current_pos(),
            )
        return Coalesce(tuple(exprs))

    # lookup := 'lookup' '(' expression ',' STRING ')'
    def _lookup_expr(self) -> Lookup:
        self._expect(TT.IDENT, hint="Expected 'lookup'")
        self._expect(TT.LPAREN, hint="Expected '(' after 'lookup'")
        expr = self._expression()
        self._expect(TT.COMMA, hint="Expected ',' after expression in lookup()")
        table_tok = self._expect(TT.STRING, hint="lookup() table name must be a string literal")
        self._expect(TT.RPAREN, hint="Expected ')' to close 'lookup(...'")
        return Lookup(expr, str(table_tok.value))

    # custom_lookup := 'custom_lookup' '(' expression ',' map_literal (',' STRING)? ')'
    def _custom_lookup_expr(self) -> CustomLookup:
        self._expect(TT.IDENT, hint="Expected 'custom_lookup'")
        self._expect(TT.LPAREN, hint="Expected '(' after 'custom_lookup'")
        expr = self._expression()
        self._expect(TT.COMMA, hint="Expected ',' after expression in custom_lookup()")
        mapping = self._map_literal()
        if not mapping:
            raise DSLSyntaxError(
                "custom_lookup() mapping must not be empty",
                expression=self._expr,
                position=self._current_pos(),
            )
        # Optional: , "base_table_name"
        base_table: str | None = None
        if self._match(TT.COMMA):
            self._advance()
            tbl_tok = self._expect(TT.STRING, hint="custom_lookup() base table name must be a string literal")
            base_table = str(tbl_tok.value)
        self._expect(TT.RPAREN, hint="Expected ')' to close 'custom_lookup(...'")
        return CustomLookup(expr, mapping, base_table)

    # map_literal := '{' (literal ':' literal (',' literal ':' literal)* ','?)? '}'
    def _map_literal(self) -> tuple[tuple[Literal, Literal], ...]:
        self._expect(TT.LBRACE, hint="Expected '{' to start mapping")
        pairs: list[tuple[Literal, Literal]] = []
        while not self._match(TT.RBRACE):
            if pairs:
                self._expect(TT.COMMA, hint="Expected ',' between mapping entries")
            if self._match(TT.RBRACE):  # trailing comma
                break
            key = self._literal_atom()
            self._expect(TT.COLON, hint="Expected ':' between key and value")
            val = self._literal_atom()
            pairs.append((key, val))
        self._expect(TT.RBRACE, hint="Expected '}' to close mapping")
        return tuple(pairs)

    def _literal_atom(self) -> Literal:
        """Parse a scalar literal (string, number, bool, null) for map keys/values."""
        tok = self._peek()
        if tok.type == TT.NUMBER:
            self._advance()
            return Literal(tok.value)
        if tok.type == TT.STRING:
            self._advance()
            return Literal(tok.value)
        if tok.type == TT.IDENT and str(tok.value) in ("true", "false", "null"):
            self._advance()
            if str(tok.value) == "true":
                return Literal(True)
            if str(tok.value) == "false":
                return Literal(False)
            return Literal(None)
        raise DSLSyntaxError(
            f"Expected a literal value (string, number, true, false, null), got {tok.value!r}",
            expression=self._expr,
            position=tok.pos,
        )

    # args := expression (',' expression)*
    def _args(self) -> list[ASTNode]:
        args: list[ASTNode] = [self._expression()]
        while self._match(TT.COMMA):
            self._advance()
            args.append(self._expression())
        return args


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_dsl(expression: str) -> ASTNode:
    """Parse *expression* and return the root AST node.

    Raises :class:`schemashift.errors.DSLSyntaxError` on any syntax error.
    """
    if not expression or not expression.strip():
        raise DSLSyntaxError(
            "Expression must not be empty",
            expression=expression,
            position=0,
        )
    return _Parser(expression).parse()
