# LLM-assisted config generation

When a file arrives from a source you have no config for, `smart_transform()` can generate one automatically using a language model.

## How it works

1. schemashift reads the file headers and a small sample (default 5 rows)
2. Sends them to your LLM along with the target schema and the DSL reference
3. The LLM returns a `FormatConfig` as JSON
4. schemashift validates the config: parses all DSL expressions, transforms the sample rows
5. On failure, retries up to N times (default 2) with the error appended to the prompt
6. On success, optionally saves the config to the registry

## Installation

```bash
pip install "schemashift[llm]"
```

## Supported LLM providers

schemashift accepts any LangChain `BaseChatModel`. Two providers are tested and supported:

::::{tab-set}

:::{tab-item} Anthropic (direct)
```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
```

Set `ANTHROPIC_API_KEY` in your environment or `.env` file.
:::

:::{tab-item} Azure AI Foundry
```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-haiku-4-5",
    api_key="<FOUNDRY_API_KEY>",
    base_url="https://<resource>.services.ai.azure.com/anthropic",
)
```

Or via environment variables (`FOUNDRY_API_KEY` + `FOUNDRY_RESOURCE`).
:::

::::

## Basic usage

```python
import schemashift as ss
from langchain_anthropic import ChatAnthropic

schema = ss.TargetSchema.from_yaml("schemas/lot_movement.yaml")
registry = ss.FileSystemRegistry("./configs/")
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

result = ss.smart_transform(
    "sap_erp.csv",
    registry=registry,
    target_schema=schema,
    llm=llm,
    auto_register=True,  # saves the config so next run hits the registry
)
```

If the file matches an existing config in the registry, `smart_transform()` uses it directly — the LLM is only called on a miss.

## Human review step

Pass a `review_fn` to inspect and optionally edit the generated config before it's applied:

```python
def review(config: ss.FormatConfig, sample_df) -> ss.FormatConfig | None:
    print(config.model_dump_json(indent=2))
    print(sample_df)
    # return config to accept, return None to reject
    return config

result = ss.smart_transform(
    "sap_erp.csv",
    registry=registry,
    target_schema=schema,
    llm=llm,
    review_fn=review,
    auto_register=True,
)
```

## Generating a config without transforming

If you want only the config (e.g. to store it or inspect it before use):

```python
from schemashift.llm import generate_config

config = generate_config(
    path="sap_erp.csv",
    target_schema=schema,
    llm=llm,
    max_retries=3,
)
registry.register(config)
```

## Error handling

```python
from schemashift.errors import LLMGenerationError

try:
    result = ss.smart_transform(...)
except LLMGenerationError as e:
    print(f"Failed after {len(e.attempts)} attempts")
    for i, attempt in enumerate(e.attempts):
        print(f"Attempt {i+1}: {attempt}")
```

`LLMGenerationError.attempts` contains the error message from each retry, useful for debugging prompt issues.

## CLI

```bash
# Generate a config and print it
schemashift generate data.csv --target-schema schemas/lot_movement.yaml

# Generate and save to the registry
schemashift generate data.csv \
    --registry ./configs/ \
    --target-schema schemas/lot_movement.yaml

# Generate with interactive review before saving
schemashift generate data.csv \
    --registry ./configs/ \
    --target-schema schemas/lot_movement.yaml \
    --interactive
```

### Credential auto-detection

The CLI loads a `.env` file from the working directory and picks the provider automatically:

| Priority | Variables | Provider |
|----------|-----------|----------|
| 1 | `FOUNDRY_API_KEY` + `FOUNDRY_ENDPOINT` | Azure AI Foundry |
| 2 | `ANTHROPIC_API_KEY` | Anthropic |

If `FOUNDRY_RESOURCE` is set without `FOUNDRY_ENDPOINT`, the endpoint is inferred as
`https://<FOUNDRY_RESOURCE>.services.ai.azure.com/api/projects/<FOUNDRY_RESOURCE>`.
