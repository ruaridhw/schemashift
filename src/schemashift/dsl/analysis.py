"""AST analysis helpers for the schemashift DSL."""

from schemashift.dsl.ast_nodes import (
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


def collect_col_refs(node: ASTNode) -> set[str]:
    """Collect every column name referenced by ``col(...)`` calls in an AST."""
    refs: set[str] = set()
    _visit(node, refs)
    return refs


def _visit(node: ASTNode, refs: set[str]) -> None:
    match node:
        case ColRef(name=name):
            refs.add(name)
        case Literal():
            return
        case BinaryOp(left=left, right=right):
            _visit(left, refs)
            _visit(right, refs)
        case UnaryOp(operand=operand):
            _visit(operand, refs)
        case MethodCall(obj=obj, args=args):
            _visit(obj, refs)
            for arg in args:
                _visit(arg, refs)
        case WhenClause(condition=condition, value=value):
            _visit(condition, refs)
            _visit(value, refs)
        case WhenChain(whens=whens, otherwise=otherwise):
            for when in whens:
                _visit(when, refs)
            _visit(otherwise, refs)
        case Coalesce(exprs=exprs):
            for expr in exprs:
                _visit(expr, refs)
        case Lookup(expr=expr):
            _visit(expr, refs)
        case CustomLookup(expr=expr):
            _visit(expr, refs)
