"""Add splitting and asymmetric-error evidence using saved data only.

No model loading, inference, fitting, tuning or new evaluation is performed.
"""
from pathlib import Path
import base64
import gzip
import hashlib
import io
import json
import struct

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FINAL = ROOT / "results/final-evaluation-v1"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def idx(path):
    raw = gzip.decompress(path.read_bytes())
    a, b, dtype, dims = struct.unpack(">BBBB", raw[:4])
    assert (a, b, dtype) == (0, 0, 8) and dims in (1, 3)
    offset = 4 + dims * 4
    shape = struct.unpack(">" + "I" * dims, raw[4:offset])
    return np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape(shape)

def png(image):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

data = json.loads((HERE / "data.json").read_text())
classes = data["classes"]
meta = json.loads((ROOT / "data/splits/train-validation-v1.json").read_text())
evaluation = json.loads((FINAL / "run-metadata.json").read_text())
split_path = ROOT / "data/splits/train-validation-v1.npz"
assert sha(split_path) == meta["split_file_sha256"] == evaluation["split_sha256"]
for record in evaluation["training_sources"] + evaluation["test_sources"]:
    assert sha(ROOT / "data/raw" / record["filename"]) == record["sha256"]
shared = {}
for folder in ["logistic-regression-regularization", "knn-comparison", "cnn-improvements"]:
    record = json.loads((ROOT / "results" / folder / "run-metadata.json").read_text())
    shared[folder] = record["split_sha256"] == meta["split_file_sha256"]
assert all(shared.values())
with np.load(split_path, allow_pickle=False) as split:
    train_ids, validation_ids = split["train_indices"], split["validation_indices"]
labels = idx(ROOT / "data/raw/train-labels-idx1-ubyte.gz")
test_labels = idx(ROOT / "data/raw/t10k-labels-idx1-ubyte.gz")
class_counts = pd.read_csv(FINAL / "class-counts.csv")
subsets = []
for name, values, field in [("Training", labels[train_ids], "Training"), ("Validation", labels[validation_ids], "Validation"), ("Official test", test_labels, "Test")]:
    counts = np.bincount(values, minlength=10)
    np.testing.assert_array_equal(counts, class_counts[field])
    assert np.all(counts / len(values) == .1)
    subsets.append({"name": name, "total": len(values), "per_class": counts.tolist(), "proportions": (counts / len(values)).tolist()})
data["split_details"] = {"subsets": subsets, "splitter": meta["splitter"], "n_splits": meta["n_splits"],
                         "validation_fold": meta["selected_fold"], "training_folds": [1, 2, 3, 4],
                         "shuffle": True, "seed": meta["seed"], "group_definition": meta["group_definition"],
                         "duplicate_groups": meta["duplicate_audit"]["training_duplicate_groups"],
                         "all_models_share_saved_split": all(shared.values()), "split_sha256": sha(split_path),
                         "test_origin": "Official publisher-provided Fashion-MNIST test set; its balance was checked, not constructed"}

predictions = pd.read_csv(FINAL / "predictions.csv")
np.testing.assert_array_equal(predictions.test_index, np.arange(10000))
np.testing.assert_array_equal(predictions.actual, test_labels)
matrix = pd.read_csv(FINAL / "cnn-confusion-counts.csv", index_col=0)
computed = np.zeros((10, 10), dtype=int)
np.add.at(computed, (predictions.actual.to_numpy(), predictions.CNN.to_numpy()), 1)
np.testing.assert_array_equal(computed, matrix.to_numpy())
np.testing.assert_array_equal(matrix.to_numpy(), data["cnn_matrix"])
images = idx(ROOT / "data/raw/t10k-images-idx3-ubyte.gz")
shirt = classes.index("Shirt")
pairs = []
for other in ["T-shirt/top", "Coat", "Pullover"]:
    other_id = classes.index(other)
    examples = []
    for actual, predicted in [(shirt, other_id), (other_id, shirt)]:
        matches = predictions[(predictions.actual == actual) & (predictions.CNN == predicted)]
        count = int(matrix.iloc[actual, predicted])
        assert len(matches) == count and count > 0
        source_id = int(matches.test_index.iloc[0])
        examples.append({"id": source_id, "actual": classes[actual], "predicted": classes[predicted],
                         "count": count, "image": png(images[source_id]), "true_class_support": int(matrix.iloc[actual].sum())})
    pairs.append({"other_class": other, "forward": examples[0], "reverse": examples[1]})
shirt_row = matrix.iloc[shirt]
data["shirt_analysis"] = {"correct": int(shirt_row.iloc[shirt]), "support": int(shirt_row.sum()),
                          "recall": float(shirt_row.iloc[shirt] / shirt_row.sum()), "pairs": pairs,
                          "example_selection": "First official test index in the saved CNN predictions matching each direction; illustrative, not a random sample",
                          "source": "results/final-evaluation-v1/cnn-confusion-counts.csv and predictions.csv"}
assert data["shirt_analysis"]["correct"] == 676 and data["shirt_analysis"]["support"] == 1000

data["scene_notes"] = json.loads((HERE / "scene-notes.json").read_text())
(HERE / "data.json").write_text(json.dumps(data, separators=(",", ":")))
source_paths = ["notebooks/01-data-exploration.ipynb", "data/splits/train-validation-v1.json", "data/splits/train-validation-v1.npz",
                "results/final-evaluation-v1/cnn-confusion-counts.csv", "results/final-evaluation-v1/predictions.csv", "results/final-evaluation-v1/class-counts.csv"]
manifest = {"operation": "Saved-evidence extension only; no training, inference or tuning", "split_details": data["split_details"],
            "shirt_recall": data["shirt_analysis"]["recall"], "pairs": [{"other_class": p["other_class"], "forward_count": p["forward"]["count"],
            "reverse_count": p["reverse"]["count"], "forward_example_index": p["forward"]["id"], "reverse_example_index": p["reverse"]["id"]} for p in pairs],
            "example_selection": data["shirt_analysis"]["example_selection"], "source_sha256": {p: sha(ROOT / p) for p in source_paths}}
(HERE / "evidence-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"split_counts": subsets, "shirt_pairs": manifest["pairs"], "operation": manifest["operation"]}, indent=2))
