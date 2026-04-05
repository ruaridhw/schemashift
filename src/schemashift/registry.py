"""Format config registries — in-memory and file-system backed."""

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

from schemashift.errors import ConfigValidationError
from schemashift.models import FormatConfig
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

    def load_schema(self, name: str | None = None) -> "TargetSchema | None":
        """Load a target schema associated with this registry, if supported."""
        return None


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

    def __init__(self, path: Path) -> None:
        self._path = path
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
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                configs.append(FormatConfig.model_validate(data))
            except Exception as exc:
                raise ConfigValidationError(f"Failed to load config from '{p}': {exc}") from exc
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
        exists, returns it. Returns ``None`` if the ``schemas/`` directory
        does not exist, is empty, or the named file is not found.

        Raises:
            ValueError: If multiple schemas exist and no explicit name is given.
        """
        path = _resolve_schema_path(self._path / "schemas", name)
        if path is None:
            return None
        return TargetSchema.from_yaml(path)


def _resolve_schema_path(schemas_dir: Path, name: str | None = None) -> Path | None:
    """Resolve a schema file path inside a ``schemas`` directory."""
    if not schemas_dir.exists():
        return None

    if name is not None:
        for suffix in (".yaml", ".yml"):
            path = schemas_dir / f"{name}{suffix}"
            if path.exists():
                return path
        return None

    yamls = sorted(schemas_dir.glob("*.yaml")) + sorted(schemas_dir.glob("*.yml"))
    if not yamls:
        return None
    if len(yamls) > 1:
        names = [path.name for path in yamls]
        raise ValueError(f"Multiple schemas found in '{schemas_dir}': {names}. Use an explicit schema name.")
    return yamls[0]
