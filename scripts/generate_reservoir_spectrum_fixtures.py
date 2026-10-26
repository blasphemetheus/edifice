"""Regenerate independent NumPy eigvals references for the host QR solver."""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(471)
cases = []
for size in [3, 8, 32, 64]:
    for sparsity in [0, 0.5, 0.9]:
        matrix = rng.normal(size=(size, size)) * (rng.random((size, size)) > sparsity)
        cases.append({"matrix": matrix.tolist(), "radius": float(max(abs(np.linalg.eigvals(matrix))))})

Path("test/fixtures/reservoir_spectral_radius.json").write_text(json.dumps(cases) + "\n")
