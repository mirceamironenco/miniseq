import importlib
import sys
import threading
import types
from typing import Any

_lock = threading.RLock()


def soft_lazy_import(fullname: str) -> types.ModuleType:
    existing = sys.modules.get(fullname)
    if existing is not None:
        return existing
    return _SoftLazyModule(fullname)


def lazy_import(fullname: str) -> types.ModuleType:
    """
    Return a module proxy whose module body executes only on first attribute access.
    If the module is already in sys.modules, it is returned as-is.
    """
    mod = sys.modules.get(fullname)
    if mod is not None:
        return mod

    with _lock:
        mod = sys.modules.get(fullname)
        if mod is not None:
            return mod

        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            # LazyLoader can't wrap namespace packages (loader=None)
            raise ModuleNotFoundError(
                f"No module named {fullname!r} (no executable loader)"
            )

        # Wrap the loader to defer execution until first attribute access
        spec.loader = importlib.util.LazyLoader(spec.loader)

        module = importlib.util.module_from_spec(spec)
        # Register before exec_module to support circular imports
        sys.modules[fullname] = module
        spec.loader.exec_module(
            module
        )  # this installs lazy machinery; body not run yet

        _bind_on_parent(fullname, module)
        return module


def _bind_on_parent(fullname: str, module: types.ModuleType) -> None:
    parent, _, child = fullname.rpartition(".")
    if not parent:
        return
    # Ensure parent is imported and bind the attribute
    parent_mod = sys.modules.get(parent) or importlib.import_module(parent)
    if getattr(parent_mod, child, None) is not module:
        setattr(parent_mod, child, module)


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


class _SoftLazyModule(types.ModuleType):
    def __init__(self, fullname: str, err_msg: str | None = None) -> None:
        super().__init__(fullname)
        object.__setattr__(self, "_fullname", fullname)
        object.__setattr__(self, "_err_msg", err_msg)
        object.__setattr__(self, "_loaded", False)

    def _load(self) -> types.ModuleType:
        if object.__getattribute__(self, "_loaded"):
            return sys.modules.get(object.__getattribute__(self, "_fullname"), self)

        with _lock:
            if object.__getattribute__(self, "_loaded"):
                return sys.modules.get(object.__getattribute__(self, "_fullname"), self)

            fullname = object.__getattribute__(self, "_fullname")

            # Adopt if someone imported it eagerly meanwhile
            existing = sys.modules.get(fullname)
            if existing is not None and existing is not self:
                self.__dict__.update(existing.__dict__)
                object.__setattr__(self, "_loaded", True)
                return existing

            # Import the real module now
            try:
                mod = importlib.import_module(fullname)
            except ModuleNotFoundError as e:
                err_msg = object.__getattribute__(self, "_err_msg") or (
                    f"Optional dependency {fullname!r} is required for this feature."
                )
                raise ModuleNotFoundError(err_msg) from e

            # Morph in place without clearing private attrs
            self.__dict__.update(mod.__dict__)
            sys.modules[fullname] = mod
            object.__setattr__(self, "_loaded", True)

            # (optional) bind onto parent package for pkg.sub attr access
            parent, _, child = fullname.rpartition(".")
            if parent:
                parent_mod = sys.modules.get(parent)
                if (
                    parent_mod is not None
                    and getattr(parent_mod, child, None) is not mod
                ):
                    setattr(parent_mod, child, mod)

            return mod

    def __getattr__(self, name: str) -> Any:
        # Avoid delegating private/internal names
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._load(), name)
