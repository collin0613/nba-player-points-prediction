"""Run in a Snowflake notebook after uploading the four bundle files to ARTIFACTS."""
import hashlib
import json
import platform
import warnings
from pathlib import Path
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from snowflake.snowpark.context import get_active_session

session = get_active_session()
stage = "@NBA_ML_DEV.ML.ARTIFACTS"
folder = Path("/tmp/nba_ml_compatibility")
folder.mkdir(parents=True, exist_ok=True)
stem = "points_population_2026-27_role_v2_20261109"
names = ["manifest.json", "parity_fixture.json", f"{stem}.json", f"{stem}.joblib"]
for name in names:
    session.file.get(f"{stage}/{name}", str(folder))

manifest = json.loads((folder / "manifest.json").read_text())
for name, expected_hash in manifest["sha256"].items():
    actual_hash = hashlib.sha256((folder / name).read_bytes()).hexdigest()
    assert actual_hash == expected_hash, f"File checksum mismatch: {name}"
print("File checksums: PASS")
print("Cloud Python:", platform.python_version())
for package, local_version in manifest["packages"].items():
    print(f"{package}: local={local_version}, cloud={version(package)}")

fixture = json.loads((folder / "parity_fixture.json").read_text())
x = pd.DataFrame(fixture["rows"], columns=fixture["features"])
for column in fixture["features"]:
    x[column] = (x[column].astype(str) if column == "player_key"
                 else pd.to_numeric(x[column], errors="raise").astype("float64"))

# Only load the project's own trusted artifact, after checking its bytes above.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    model = joblib.load(folder / f"{stem}.joblib")
    actual_raw = np.asarray(model.predict(x), dtype=float)
for warning in caught:
    print(f"WARNING ({warning.category.__name__}): {warning.message}")

expected_raw = np.asarray(fixture["expected_raw_predictions"], dtype=float)
assert actual_raw.shape == expected_raw.shape, "Prediction shape mismatch"
assert np.isfinite(actual_raw).all(), "Nonfinite predictions"
np.testing.assert_allclose(actual_raw, expected_raw,
                           atol=fixture["atol"], rtol=fixture["rtol"])
np.testing.assert_allclose(np.maximum(0, actual_raw), fixture["expected_points"],
                           atol=fixture["atol"], rtol=fixture["rtol"])
print("Rows tested:", len(x))
print("Maximum absolute difference:", float(np.max(np.abs(actual_raw - expected_raw))))
print("INFERENCE PARITY: PASS")
print("This checks fixed-input inference only; cloud feature generation and production deployment are not yet validated.")
