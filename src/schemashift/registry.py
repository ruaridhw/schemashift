"""Format config registries — in-memory and file-system backed."""

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from schemashift.errors import ConfigValidationError
from schemashift.models import FormatConfig

if TYPE_CHECKING:
    from schemashift.target_schema import TargetSchema


class Registry(ABC):
    """Abstract base registry for FormatConfig objects."""

    @abstractmethod
    def get(self, name: str) -> FormatConfig | None:
        """Return the config with the given name, or None if not found."""

    @abstractmethod
    def register(self, config: FormatConfig) -> None:
        """Store a config, replacing any existing config with the same name."""

    @abstractmethod
    def list_configs(self) -> list[FormatConfig]:
        """Return all registered configs."""

    @abstractmethod
    def delete(self, name: str) -> bool:
        """Remove the config with the given name.

        Returns True if the config existed and was removed, False otherwise.
        """


class DictRegistry(Registry):
    """In-memory registry suitable for testing and embedded use."""

    def __init__(self) -> None:
        self._configs: dict[str, FormatConfig] = {}

    def get(self, name: str) -> FormatConfig | None:
        return self._configs.get(name)

    def register(self, config: FormatConfig) -> None:
        self._configs[config.name] = config

    def list_configs(self) -> list[FormatConfig]:
        return list(self._configs.values())

    def delete(self, name: str) -> bool:
        if name in self._configs:
            del self._configs[name]
            return True
        return False


class FileSystemRegistry(Registry):
    """Stores configs as JSON files under a directory.

    Each config is persisted as ``{directory}/{name}.json``.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)

    def _config_path(self, name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
            raise ConfigValidationError(
                f"Invalid config name {name!r}: only letters, digits, underscores, and hyphens are allowed."
            )
        return self._path / f"{name}.json"

    def get(self, name: str) -> FormatConfig | None:
        p = self._config_path(name)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return FormatConfig.model_validate(data)

    def register(self, config: FormatConfig) -> None:
        p = self._config_path(config.name)
        p.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    def list_configs(self) -> list[FormatConfig]:
        configs: list[FormatConfig] = []
        for p in sorted(self._path.glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            configs.append(FormatConfig.model_validate(data))
        return configs

    def delete(self, name: str) -> bool:
        p = self._config_path(name)
        if p.exists():
            p.unlink()
            return True
        return False

    def load_schema(self, name: str | None = None) -> "TargetSchema | None":
        """Load a TargetSchema from the schemas/ subdirectory.

        If *name* is provided, loads ``{schemas_dir}/{name}.yaml`` (falling
        back to ``.yml``).  If *name* is ``None`` and exactly one schema file
        exists, returns it.  Returns ``None`` if the ``schemas/`` directory
        does not exist, is empty, or the named file is not found.
        """
        from schemashift.target_schema import TargetSchema

        schemas_dir = self._path / "schemas"
        if not schemas_dir.exists():
            return None

        if name is not None:
            path = schemas_dir / f"{name}.yaml"
            if path.exists():
                return TargetSchema.from_yaml(path)
            path = schemas_dir / f"{name}.yml"
            if path.exists():
                return TargetSchema.from_yaml(path)
            return None

        yamls = list(schemas_dir.glob("*.yaml")) + list(schemas_dir.glob("*.yml"))
        if len(yamls) == 1:
            return TargetSchema.from_yaml(yamls[0])
        return None
