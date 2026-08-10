"""Re-exports this paper's 3 metrics from the core pqc_diagnostics library.

kernel_usefulness (KTA + SVM accuracy) also lives in pqc_diagnostics.metrics
but is deliberately NOT re-exported/used here - it wasn't feeding any poster
plot, and in the noisy condition its fidelity-matrix Aer calls were roughly
half the per-seed cost. See pqc_diagnostics.metrics.kernel_usefulness if
you want it for your own PQC.
"""

from __future__ import annotations

from pqc_diagnostics.metrics import (  # noqa: F401
    Simulate,
    State,
    expressibility_kl,
    gradient_variance,
    meyer_wallach,
)
