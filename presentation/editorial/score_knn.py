"""Measure leave-one-out accuracy of the frozen three-neighbour classifier."""
from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from final_evaluation import FILES, read_idx


def main():
    name, fingerprint = FILES["kNN"]
    checkpoint = ROOT / "results" / name
    with checkpoint.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == fingerprint
    model = joblib.load(checkpoint)
    ids = np.load(ROOT / "data/splits/train-validation-v1.npz")["train_indices"]
    labels = read_idx(ROOT / "data/raw/train-labels-idx1-ubyte.gz")[ids]
    assert model.n_samples_fit_ == len(ids) == 48000 and model.n_neighbors == 3
    assert model.weights == "uniform"
    with threadpool_limits(limits=4):
        # An omitted query uses the fitted observations and excludes each own index.
        neighbours = model.kneighbors(return_distance=False)
    assert not np.any(neighbours == np.arange(len(ids))[:, None])
    votes = labels[neighbours]
    predicted = np.array([np.bincount(row, minlength=10).argmax() for row in votes])
    correct = int(np.sum(labels == predicted))
    result = dict(correct=correct, total=len(ids), accuracy=correct / len(ids),
                  checkpoint_sha256=fingerprint, k=3, self_excluded=True)
    (HERE / "knn-training.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
