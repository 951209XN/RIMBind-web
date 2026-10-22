"""Keep implementation equivalence separate from historical GPU portability.

This report alone never authorizes release or replaces an earlier failed gate.
"""
import numpy as np


def compare_probabilities(candidate, reference, historical, *, candidate_runtime, reference_runtime):
    if not candidate_runtime or candidate_runtime != reference_runtime:
        raise ValueError('Same-runtime comparison requires identical explicit runtime fingerprints')
    arrays = [np.asarray(x, dtype=np.float64) for x in (candidate, reference, historical)]
    if not all(x.ndim == 1 and x.shape == arrays[0].shape and x.size > 0 and
               np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all() for x in arrays):
        raise ValueError('Three finite, aligned probability vectors are required')
    implementation_error = float(np.abs(arrays[0]-arrays[1]).max())
    historical_error = float(np.abs(arrays[0]-arrays[2]).max())
    return dict(same_runtime_tolerance=1e-5, historical_tolerance=1e-5,
                same_runtime_max_probability_abs_error=implementation_error,
                same_runtime_equivalence_passed=bool(implementation_error < 1e-5),
                historical_max_probability_abs_error=historical_error,
                historical_equivalence_passed=bool(historical_error < 1e-5),
                release_authorized=False, policy_change_approved=False,
                notice='Independent diagnostic fields, not an automatic release decision.')
