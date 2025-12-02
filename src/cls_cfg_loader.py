
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Iterator

class DictWrapper:
    """Wrapper class that converts dictionaries to objects with attribute access.
    
    Supports both attribute access (cfg.train.epochs) and dict-like access (cfg['train']['epochs']).
    Recursively wraps nested dictionaries and lists.
    """

    def __init__(self, d: Dict[str, Any]):
        """Initialize wrapper with a dictionary.
        
        Args:
            d: Dictionary to wrap.
        """
        self._data = {}
        for k, v in d.items():
            if isinstance(v, dict):
                v = DictWrapper(v)
            elif isinstance(v, list):
                v = [DictWrapper(i) if isinstance(i, dict) else i for i in v]
            self._data[k] = v
            setattr(self, k, v)

    def __getitem__(self, key: str) -> Any:
        """Enable dict-like access."""
        return self._data[key]

    def __iter__(self) -> Iterator:
        """Enable iteration over keys."""
        return iter(self._data)

    def items(self):
        """Return items as key-value pairs."""
        return self._data.items()

    def keys(self):
        """Return all keys."""
        return self._data.keys()

    def values(self):
        """Return all values."""
        return self._data.values()

    def to_dict(self) -> Dict[str, Any]:
        """Convert wrapped structure back to plain dictionary.
        
        Returns:
            Plain dictionary with all nested structures unwrapped.
        """
        def unwrap(v):
            if isinstance(v, DictWrapper):
                return v.to_dict()
            elif isinstance(v, list):
                return [unwrap(i) for i in v]
            else:
                return v
        return {k: unwrap(v) for k, v in self._data.items()}


class Config_loader:
    """Load and manage YAML configuration files with nested structure support.
    
    Automatically resolves relative paths, loads nested config files,
    and provides both attribute and dictionary access to configuration values.
    """

    def __init__(self, file_path: str):
        """Initialize config loader and load configuration from file.
        
        Args:
            file_path: Relative path to main config file from project root.
                      Example: "Data/Training/Configs/configs.yaml"
        
        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config entry format is invalid.
        """
        self.root_dir = self._get_project_root()
        self.main_cfg_path = os.path.join(self.root_dir, file_path)

        if not os.path.exists(self.main_cfg_path):
            raise FileNotFoundError(
                f"Config file not found: {self.main_cfg_path}\n"
                f"Expected path: {file_path}\n"
                f"Project root: {self.root_dir}"
            )

        with open(self.main_cfg_path, "r", encoding="utf-8") as f:
            main_cfg_raw = yaml.load(f, Loader=yaml.FullLoader) or {}

        self._process_config(main_cfg_raw)
        print("Status:✅ Config file loaded")

    # ========== Internal API ==========

    def _get_project_root(self) -> str:
        """Get project root directory (parent of src/).
        
        Returns:
            Absolute path to project root directory.
        """
        current_file = Path(__file__).resolve()
        src_dir = current_file.parent
        project_root = src_dir.parent
        return str(project_root)

    def _process_config(self, main_cfg_raw: Dict[str, Any]) -> None:
        """Process main config and load nested configurations.
        
        Args:
            main_cfg_raw: Raw configuration dictionary from main YAML file.
        
        Raises:
            ValueError: If config entry has invalid format.
        """
        for key, cfg_entry in main_cfg_raw.items():
            if key.lower() in {"pathes", "paths"} and isinstance(cfg_entry, dict):
                self._process_paths(key, cfg_entry)
            elif isinstance(cfg_entry, dict) and "path" in cfg_entry:
                self._process_dict_with_path(key, cfg_entry)
            elif isinstance(cfg_entry, dict):
                self._process_dict(key, cfg_entry)
            elif isinstance(cfg_entry, str):
                self._process_string_path(key, cfg_entry)
            else:
                raise ValueError(
                    f"[Config] Invalid entry in configs.yaml: {key} = {cfg_entry}"
                )

    def _process_paths(self, key: str, cfg_entry: Dict[str, str]) -> None:
        """Process path entries and resolve them relative to project root.
        
        Args:
            key: Configuration key name.
            cfg_entry: Dictionary containing path mappings.
        """
        resolved = {
            k: os.path.normpath(os.path.join(self.root_dir, v))
            for k, v in cfg_entry.items()
        }
        setattr(self, key, DictWrapper(resolved))

    def _process_dict_with_path(self, key: str, cfg_entry: Dict[str, Any]) -> None:
        """Process dictionary entry that references an external config file.
        
        Args:
            key: Configuration key name.
            cfg_entry: Dictionary with 'path' key pointing to external config.
        """
        sub_path = os.path.normpath(
            os.path.join(self.root_dir, cfg_entry["path"])
        )
        with open(sub_path, "r", encoding="utf-8") as subf:
            sub_cfg = yaml.load(subf, Loader=yaml.FullLoader) or {}
        meta = {k: v for k, v in cfg_entry.items() if k != "path"}
        combined = {**meta, **sub_cfg}
        setattr(self, key, DictWrapper(combined))

    def _process_dict(self, key: str, cfg_entry: Dict[str, Any]) -> None:
        """Process regular dictionary entry.
        
        Args:
            key: Configuration key name.
            cfg_entry: Dictionary to wrap and set as attribute.
        """
        setattr(self, key, DictWrapper(cfg_entry))

    def _process_string_path(self, key: str, cfg_entry: str) -> None:
        """Process string entry as path to external config file.
        
        Args:
            key: Configuration key name.
            cfg_entry: Path to external config file.
        """
        sub_path = os.path.normpath(os.path.join(self.root_dir, cfg_entry))
        with open(sub_path, "r", encoding="utf-8") as subf:
            sub_cfg = yaml.load(subf, Loader=yaml.FullLoader) or {}
        setattr(self, key, DictWrapper(sub_cfg))

    def _unwrap(self, value: Any) -> Any:
        """Recursively unwrap DictWrapper objects to plain dictionaries.
        
        Args:
            value: Value to unwrap.
            
        Returns:
            Unwrapped value as plain Python types.
        """
        if isinstance(value, DictWrapper):
            return value.to_dict()
        elif isinstance(value, dict):
            return {k: self._unwrap(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._unwrap(v) for v in value]
        else:
            return value

    # ========== Public API ==========

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to plain dictionary.
        
        Returns:
            Dictionary containing all configuration values with wrapped objects unwrapped.
        """
        return {
            key: self._unwrap(value)
            for key, value in self.__dict__.items()
            if not key.startswith('_') and key not in {'root_dir', 'main_cfg_path'}
        }


# Legacy alias for backward compatibility
SDF_config_loader = Config_loader