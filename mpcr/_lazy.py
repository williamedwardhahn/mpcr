"""Stand-ins that import a heavy or optional library the first time it is used.

`from mpcr import *` used to import wandb, nltk, cv2, flashtorch and barbar (and pip-install three
of them) on every import. These proxies keep the same names in the namespace but defer the import,
so `import mpcr` is fast, works offline, and only fails if you call something that is missing.
"""
import importlib

__all__ = ['LazyModule', 'LazyAttr']


class LazyModule:
    """Behaves like the module `name` once any attribute is read."""

    def __init__(self, name, hint=None):
        self.__dict__['_name'] = name
        self.__dict__['_hint'] = hint
        self.__dict__['_mod'] = None

    def _load(self):
        if self._mod is None:
            try:
                self.__dict__['_mod'] = importlib.import_module(self._name)
            except ImportError as e:
                hint = self._hint or f'pip install {self._name}'
                raise ImportError(f'mpcr: {self._name} is not installed ({hint})') from e
        return self._mod

    def __getattr__(self, attr):
        return getattr(self._load(), attr)

    def __setattr__(self, attr, value):
        setattr(self._load(), attr, value)

    def __dir__(self):
        return dir(self._load())

    def __repr__(self):
        state = 'loaded' if self._mod is not None else 'not loaded yet'
        return f'<lazy module {self._name!r} ({state})>'


class LazyAttr:
    """Behaves like `module.attr` (usually a class or function) once it is called or inspected."""

    def __init__(self, module, attr, hint=None, call=False):
        self._module, self._attr, self._hint, self._call = module, attr, hint, call
        self._obj = None

    def _load(self):
        if self._obj is None:
            try:
                obj = getattr(importlib.import_module(self._module), self._attr)
            except ImportError as e:
                hint = self._hint or f'pip install {self._module.split(".")[0]}'
                raise ImportError(f'mpcr: {self._module} is not installed ({hint})') from e
            self._obj = obj() if self._call else obj
        return self._obj

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, attr):
        if attr.startswith('_'):
            raise AttributeError(attr)
        return getattr(self._load(), attr)

    def __repr__(self):
        return f'<lazy {self._module}.{self._attr}>'
