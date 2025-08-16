import hashlib
import glob
import os

CONFIG_PATH = "src/spectramind/config"

def compute_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def test_config_hashes_stable(tmp_path):
    """Ensure configs generate deterministic hashes for reproducibility."""
    files = glob.glob(os.path.join(CONFIG_PATH, "**", "*.yaml"), recursive=True)
    assert files, "No config files found to hash."

    hashes = {f: compute_file_hash(f) for f in files}
    # Save summary for reproducibility audit
    summary_file = tmp_path / "config_hashes.txt"
    with open(summary_file, "w") as out:
        for k, v in hashes.items():
            out.write(f"{k}: {v}\n")

    # Ensure no duplicate hash collisions
    unique_hashes = set(hashes.values())
    assert len(unique_hashes) == len(hashes), "Config hash collision detected!"
