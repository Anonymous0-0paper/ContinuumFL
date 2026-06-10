#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  ContinuumFL — Download all datasets
#
#  Downloads each dataset into ./data/ by calling download_and_prepare()
#  directly, without running any training. Safe to re-run — already-downloaded
#  datasets are skipped automatically by the loaders.
#
#  Usage:
#    bash scripts/download_datasets.sh               # download all 5 datasets
#    bash scripts/download_datasets.sh femnist       # download one dataset
#    bash scripts/download_datasets.sh cifar100 ucihar  # download specific ones
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ┌─────────────────────────────────────────────────────────────────────────────
# │ PYTHON
# └─────────────────────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    command -v python3 &>/dev/null && PYTHON_BIN="python3" || {
        echo "❌ Neither 'python' nor 'python3' found." >&2; exit 1
    }
fi
VENV_DIR="$PROJECT_ROOT/.venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    PYTHON_BIN="$VENV_DIR/bin/python"
fi

# ┌─────────────────────────────────────────────────────────────────────────────
# │ DATASETS TO DOWNLOAD
# └─────────────────────────────────────────────────────────────────────────────
ALL_DATASETS=(ucihar femnist cifar100 shakespeare speechcommands)

if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

echo "════════════════════════════════════════════════════════════"
echo "  ContinuumFL — Dataset downloader"
echo "  Datasets : ${DATASETS[*]}"
echo "  Data dir : $PROJECT_ROOT/data/"
echo "════════════════════════════════════════════════════════════"
echo ""

# ┌─────────────────────────────────────────────────────────────────────────────
# │ DOWNLOAD EACH DATASET
# └─────────────────────────────────────────────────────────────────────────────
FAILED=0
for ds in "${DATASETS[@]}"; do
    echo "──────────────────────────────────────────────────────────"
    echo "  Downloading: $ds"
    echo "──────────────────────────────────────────────────────────"

    "$PYTHON_BIN" - <<PYEOF
import sys
sys.path.insert(0, "$PROJECT_ROOT")

from types import SimpleNamespace
from src.data.federated_dataset import FederatedDataset

config = SimpleNamespace(
    dataset_name="$ds",
    intra_zone_alpha=100,
    inter_zone_alpha=5.0,
    train_test_split=0.8,
    max_samples=-1,
    shakespeare_num_speakers=None,
)

ds_obj = FederatedDataset(config, data_dir="$PROJECT_ROOT/data")
ds_obj.download_and_prepare()
print("  ✅ $ds ready.")
PYEOF

    status=$?
    if [[ $status -ne 0 ]]; then
        echo "  ⚠️  $ds failed (exit $status)"
        FAILED=$(( FAILED + 1 ))
    fi
    echo ""
done

# ┌─────────────────────────────────────────────────────────────────────────────
# │ DONE
# └─────────────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
if [[ $FAILED -eq 0 ]]; then
    echo "  ✅ All datasets downloaded successfully."
else
    echo "  ⚠️  $FAILED dataset(s) failed — check output above."
fi
echo "  Data dir : $PROJECT_ROOT/data/"
echo "════════════════════════════════════════════════════════════"
