"""Global, mutable registry of named NoiseProvider instances - mirrors
pqc_diagnostics.registry's feature-map/ansatz registry pattern, so a user can
name and share a specific (backend, patch-width, basis-gates) configuration
across a codebase without passing the object around everywhere."""

from __future__ import annotations

from pqc_diagnostics.hardware.providers import NoiseProvider

_PROVIDERS: dict[str, NoiseProvider] = {}


def register_provider(name: str, provider: NoiseProvider, *, overwrite: bool = False) -> None:
    if not overwrite and name in _PROVIDERS:
        raise ValueError(f"noise provider '{name}' is already registered (pass overwrite=True to replace it)")
    _PROVIDERS[name] = provider


def get_provider(name: str) -> NoiseProvider:
    try:
        return _PROVIDERS[name]
    except KeyError:
        raise KeyError(f"no noise provider registered under '{name}' - registered: {tuple(_PROVIDERS)}") from None
