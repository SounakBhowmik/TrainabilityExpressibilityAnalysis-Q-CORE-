"""pqc_diagnostics: expressibility, entangling capability, trainability, and
kernel-usefulness diagnostics for parameterized quantum circuits.

General-purpose library layer - works on any PQC/feature map/ansatz a user
registers, not just the ones a specific study uses. See
qcore_expr_train/ (this repo's sibling package) for a worked example that
reproduces one paper's specific reproducible study on top of this library.
"""

from __future__ import annotations
