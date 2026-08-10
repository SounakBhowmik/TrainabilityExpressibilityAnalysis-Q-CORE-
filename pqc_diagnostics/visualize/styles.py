"""Generic style registry for plotting: color/marker/linestyle keyed by an
arbitrary (dimension, name) pair, e.g. ("ansatz_family", "rotation_only").

Replaces a closed, hardcoded style dict per category (the old approach,
which KeyError'd on any custom-registered feature map/ansatz not already in
the dict) with an open registry a user can add to. An unregistered name
falls back to matplotlib's own automatic color/marker cycling (passing
`None` for color/marker/linestyle to most matplotlib plotting calls means
"pick the next one from the property cycle"), so a custom family still
plots without a KeyError - it just won't match this project's specific
palette until you register one for it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class Style:
    color: str | None = None
    marker: str | None = None
    linestyle: str | None = None


class StyleRegistry:
    def __init__(self) -> None:
        self._styles: dict[tuple[str, str], Style] = {}

    def register(
        self,
        dimension: str,
        name: str,
        *,
        color: str | None = None,
        marker: str | None = None,
        linestyle: str | None = None,
        overwrite: bool = False,
    ) -> None:
        key = (dimension, name)
        if not overwrite and key in self._styles:
            raise ValueError(f"style for {dimension}={name!r} is already registered (pass overwrite=True to replace it)")
        self._styles[key] = Style(color=color, marker=marker, linestyle=linestyle)

    def get(self, dimension: str, name: str) -> Style:
        key = (dimension, name)
        if key not in self._styles:
            warnings.warn(
                f"no style registered for {dimension}={name!r} - falling back to matplotlib's default cycling",
                stacklevel=2,
            )
            return Style()
        return self._styles[key]


# Process-global singleton, matching pqc_diagnostics.registry's convention.
DEFAULT_STYLE_REGISTRY = StyleRegistry()
