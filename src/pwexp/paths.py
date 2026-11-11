"""Repository paths. Set PWSUM_DATA to the dataset directory (default: <repo>/data) and, optionally,
PWSUM_BIN to the Rust benchmark binary (default: <repo>/rust/target/release/pwcost)."""
import os
from pathlib import Path

PWSUM_ROOT = str(Path(__file__).resolve().parents[2])
DATA_DIR = os.environ.get("PWSUM_DATA", os.path.join(PWSUM_ROOT, "data"))
RUST_BIN = os.environ.get("PWSUM_BIN", os.path.join(PWSUM_ROOT, "rust", "target", "release", "pwcost"))
