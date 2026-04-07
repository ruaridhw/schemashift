# API reference

All public symbols are importable from the top-level `schemashift` package:

```python
import schemashift as ss

ss.transform(...)
ss.FormatConfig(...)
ss.FileSystemRegistry(...)
```

## Modules

```{toctree}
:maxdepth: 1

core
registry
models
exceptions
dsl
```

## Quick index

### Transform functions
- {py:func}`schemashift.transform`
- {py:func}`schemashift.smart_transform`
- {py:func}`schemashift.validate_config`
- {py:func}`schemashift.detect_format`
- {py:func}`schemashift.read_file`

### Config models
- {py:class}`schemashift.FormatConfig`
- {py:class}`schemashift.ColumnMapping`
- {py:class}`schemashift.ReaderConfig`
- {py:class}`schemashift.TargetSchema`

### Registries
- {py:class}`schemashift.Registry`
- {py:class}`schemashift.DictRegistry`
- {py:class}`schemashift.FileSystemRegistry`

### Exceptions
- {py:exc}`schemashift.SchemaShiftError`
- {py:exc}`schemashift.DSLSyntaxError`
- {py:exc}`schemashift.DSLRuntimeError`
- {py:exc}`schemashift.AmbiguousFormatError`
- {py:exc}`schemashift.LLMGenerationError`
- {py:exc}`schemashift.SchemaValidationError`
