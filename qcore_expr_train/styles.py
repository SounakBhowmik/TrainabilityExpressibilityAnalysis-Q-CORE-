"""This paper's exact visual encoding, registered into pqc_diagnostics's
global StyleRegistry - importing this module is what makes visualize.py's
plots use this project's specific palette instead of matplotlib's default
cycling. Fixed categorical order - do not reorder or cycle.
"""

from __future__ import annotations

from pqc_diagnostics.visualize.styles import DEFAULT_STYLE_REGISTRY as STYLES

STYLES.register("ansatz_family", "rotation_only", color="#2a78d6", linestyle="-", overwrite=True)
STYLES.register("ansatz_family", "real_amplitudes", color="#1baf7a", linestyle="--", overwrite=True)

STYLES.register("feature_map_family", "zz", marker="o", color="#eda100", overwrite=True)
STYLES.register("feature_map_family", "local_z", marker="^", color="#1baf7a", overwrite=True)
STYLES.register("feature_map_family", "cx_ry", marker="D", color="#2a78d6", overwrite=True)

STYLES.register("topology", "linear", linestyle="-", overwrite=True)
STYLES.register("topology", "circular", linestyle="--", overwrite=True)
STYLES.register("topology", "full", linestyle=":", overwrite=True)
