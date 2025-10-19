import importlib
import types
from typing import Any


class LazyModule(types.ModuleType):
    def __init__(
        self,
        local_name: str,
        parent_module_globals: dict[str, Any],
        module_name: str,
    ) -> None:
        super().__init__(module_name)

        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._module_name = module_name

    def _load(self) -> types.ModuleType:
        # Import the real module
        module = importlib.import_module(self._module_name)

        # Replace the entry in the parent's globals with the real module
        self._parent_module_globals[self._local_name] = module

        # Update module __dict__ to point to the real module's attributes
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __dir__(self) -> list[str]:
        return dir(self._load())
