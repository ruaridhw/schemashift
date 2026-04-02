# DSL internals

The DSL pipeline: string → AST → `polars.Expr`. See {doc}`../user-guide/dsl` for the expression language reference.

## Entry point

```{eval-rst}
.. autofunction:: schemashift.dsl.parse_and_compile
```

## Parser

```{eval-rst}
.. automodule:: schemashift.dsl.parser
   :members:
   :undoc-members: False
```

## Compiler

```{eval-rst}
.. autofunction:: schemashift.dsl.compiler.compile_dsl
```

## AST nodes

```{eval-rst}
.. automodule:: schemashift.dsl.ast_nodes
   :members:
   :undoc-members: False
```
