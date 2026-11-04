"""Evaluate the three frozen models once; never train or select using test scores.

Run from the project root: .venv/bin/python scripts/final_evaluation.py
Existing output directories are preserved and cause an explicit stop.
"""
from pathlib import Path
import gzip
import hashlib
import json
import platform
import struct
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/final-evaluation-v1"
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
FILES = {
    "Logistic regression": ("logistic-regression-regularization/selected-model.joblib", "a224e1ce1d4deb758910a9583cf1fcffa09d635040f2fe8ad1f2e21622533ff7"),
    "kNN": ("knn-comparison/selected-model.joblib", "ea6ecc6f3b8adb599872c9278fba56f4093849afc81c96f5de1266bca741a78a"),
    "CNN": ("cnn-improvements/selected-model.pt", "1a389bf05cfe4c5eacd30ec30bc04302bfc7b74de32cd5d1b6e9570841388c01"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_idx(path):
    payload = gzip.decompress(path.read_bytes())
    a, b, dtype, dims = struct.unpack(">BBBB", payload[:4])
    if (a, b, dtype) != (0, 0, 8) or dims not in (1, 3):
        raise ValueError(f"Invalid IDX header: {path.name}")
    offset = 4 + 4 * dims
    shape = struct.unpack(">" + "I" * dims, payload[4:offset])
    values = np.frombuffer(payload, dtype=np.uint8, offset=offset)
    if values.size != np.prod(shape):
        raise ValueError(f"Invalid IDX size: {path.name}")
    return values.reshape(shape)


def build_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
        nn.Dropout(0.25), nn.Linear(128, 10),
    )


@torch.inference_mode()
def predict(model, images, is_cnn):
    chunks = []
    with threadpool_limits(limits=4):
        for start in range(0, len(images), 256):
            batch = images[start:start + 256]
            if is_cnn:
                inputs = torch.from_numpy(batch.astype(np.float32) / 255).unsqueeze(1)
                chunks.append(model(inputs).softmax(dim=1).numpy())
            else:
                inputs = batch.reshape(len(batch), -1).astype(np.float64) / 255
                chunks.append(model.predict_proba(inputs))
    probabilities = np.concatenate(chunks)
    assert probabilities.shape == (len(images), 10)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    return probabilities.argmax(axis=1), probabilities


def metrics(name, split, truth, predicted):
    report = classification_report(truth, predicted, labels=range(10), output_dict=True, zero_division=0)
    summary = {"model": name, "split": split, "n": len(truth), "correct": int((truth == predicted).sum()),
               "accuracy": accuracy_score(truth, predicted),
               **{f"macro_{key}": report["macro avg"][key] for key in ("precision", "recall", "f1-score")},
               "shirt_recall": report["6"]["recall"]}
    rows = [{"model": name, "split": split, "class_id": i, "class": label, **report[str(i)]}
            for i, label in enumerate(CLASSES)]
    return summary, rows


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserving existing evaluation: {OUT}. Read its saved outputs; do not rerun for selection.")
    torch.set_num_threads(4)
    split_path = ROOT / "data/splits/train-validation-v1.npz"
    split_meta = json.loads((split_path.with_suffix('.json')).read_text())
    assert sha(split_path) == split_meta["split_file_sha256"]
    assert sha(split_path) == "e3942a2e625003307207e88c93a7c8b81bbcb7db54f0f1dfe79382352fc1e729"
    for record in split_meta["files"]:
        assert sha(ROOT / "data/raw" / record["filename"]) == record["sha256"]
    models = {}
    for name, (relative, expected) in FILES.items():
        path = ROOT / "results" / relative
        assert sha(path) == expected, f"Frozen checkpoint changed: {name}"
        if name == "CNN":
            model = build_cnn()
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
            assert sum(p.numel() for p in model.parameters()) == 421642
        else:
            model = joblib.load(path)
            np.testing.assert_array_equal(model.classes_, np.arange(10))
        models[name] = model
    assert models["Logistic regression"].C == 0.1
    assert models["kNN"].n_neighbors == 3
    images = read_idx(ROOT / "data/raw/train-images-idx3-ubyte.gz")
    labels = read_idx(ROOT / "data/raw/train-labels-idx1-ubyte.gz")
    with np.load(split_path, allow_pickle=False) as split:
        train_ids, val_ids = split["train_indices"], split["validation_indices"]
    assert images.shape == (60000, 28, 28) and labels.shape == (60000,)
    assert len(train_ids) == 48000 and len(val_ids) == 12000
    np.testing.assert_array_equal(np.sort(np.r_[train_ids, val_ids]), np.arange(60000))
    saved = pd.read_csv(ROOT / "results/validation-comparison/aligned-validation-predictions.csv")
    np.testing.assert_array_equal(saved.source_index, val_ids)
    np.testing.assert_array_equal(saved.actual, labels[val_ids])
    summary_rows, class_rows, parity = [], [], {}
    for name, model in models.items():
        predictions, _ = predict(model, images[val_ids], name == "CNN")
        disagreements = int((predictions != saved[name].to_numpy()).sum())
        assert disagreements == 0, f"CPU validation prediction drift: {name}"
        parity[name] = disagreements
        summary, rows = metrics(name, "Validation", labels[val_ids], predictions)
        summary_rows.append(summary)
        class_rows.extend(rows)
        print(f"Validation parity passed: {name}", flush=True)

    # Test data are opened only after provenance and validation parity pass.
    test_sources = []
    for filename, expected_md5 in [
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]:
        path = ROOT / "data/raw" / filename
        assert hashlib.md5(path.read_bytes()).hexdigest() == expected_md5
        test_sources.append({"filename": filename, "md5_verified": True, "sha256": sha(path),
                             "source": f"https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/{split_meta['source_commit']}/data/fashion/{filename}"})
    test_images = read_idx(ROOT / "data/raw/t10k-images-idx3-ubyte.gz")
    test_labels = read_idx(ROOT / "data/raw/t10k-labels-idx1-ubyte.gz")
    assert test_images.shape == (10000, 28, 28) and test_labels.shape == (10000,)
    assert np.isin(test_labels, range(10)).all()
    development_hashes = {hashlib.sha256(im.tobytes()).digest() for im in images}
    overlap = np.array([hashlib.sha256(im.tobytes()).digest() in development_hashes for im in test_images])
    OUT.mkdir(parents=True, exist_ok=False)
    predictions_table = pd.DataFrame({"test_index": np.arange(10000), "actual": test_labels,
                                      "overlaps_development": overlap})
    sampled_rows, confusions, matrices = [], [], {}
    for name, model in models.items():
        predicted, probabilities = predict(model, test_images, name == "CNN")
        predictions_table[name] = predicted
        stem = {"Logistic regression": "logistic-regression", "kNN": "knn", "CNN": "cnn"}[name]
        np.savez_compressed(OUT / f"{stem}-probabilities.npz", test_index=np.arange(10000), probabilities=probabilities)
        summary, rows = metrics(name, "Test", test_labels, predicted)
        summary_rows.append(summary)
        class_rows.extend(rows)
        if overlap.any():
            summary, rows = metrics(name, "Test overlap-free sensitivity", test_labels[~overlap], predicted[~overlap])
            summary_rows.append(summary)
            class_rows.extend(rows)
        cm = confusion_matrix(test_labels, predicted, labels=range(10))
        matrices[name] = cm
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(OUT / f"{stem}-confusion-counts.csv")
        pd.DataFrame(cm / cm.sum(axis=1, keepdims=True), index=CLASSES, columns=CLASSES).to_csv(OUT / f"{stem}-confusion-normalized.csv")
        for actual in range(10):
            for pred in range(10):
                if actual != pred and cm[actual, pred]:
                    confusions.append({"model": name, "actual": CLASSES[actual], "predicted": CLASSES[pred], "count": int(cm[actual, pred])})
        error_ids = np.flatnonzero(predicted != test_labels)
        chosen = np.random.default_rng(42).choice(error_ids, min(12, len(error_ids)), replace=False)
        sampled_rows.extend({"model": name, "test_index": int(i), "actual": CLASSES[test_labels[i]], "predicted": CLASSES[predicted[i]]} for i in chosen)
        print(f"Test evaluation saved: {name}: {accuracy_score(test_labels, predicted):.4%}", flush=True)
    pd.DataFrame(summary_rows).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT / "per-class-metrics.csv", index=False)
    predictions_table.to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(sampled_rows).to_csv(OUT / "sampled-errors.csv", index=False)
    pd.DataFrame(confusions).sort_values("count", ascending=False).to_csv(OUT / "confusions.csv", index=False)
    pd.DataFrame({"class": CLASSES, "Training": np.bincount(labels[train_ids], minlength=10),
                  "Validation": np.bincount(labels[val_ids], minlength=10), "Test": np.bincount(test_labels, minlength=10)}).to_csv(OUT / "class-counts.csv", index=False)
    metadata = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "test_evaluated": True,
                "primary_model_preselected": "CNN", "refit_performed": False, "protocol_sha256": sha(ROOT / "docs/final-evaluation-protocol.md"),
                "script_sha256": sha(Path(__file__)), "split_sha256": sha(split_path),
                "model_sha256": {name: expected for name, (_, expected) in FILES.items()},
                "training_sources": split_meta["files"], "test_sources": test_sources,
                "validation_prediction_disagreements": parity, "test_rows_overlapping_development": int(overlap.sum()),
                "overlap_free_test_count": int((~overlap).sum()), "test_count": len(test_labels),
                "execution": {"device": "cpu", "threads": 4, "batch_size": 256},
                "environment": {"python": platform.python_version(), **{p: version(p) for p in ["numpy", "pandas", "torch", "scikit-learn", "joblib"]}}}
    metadata["output_sha256"] = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    (OUT / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Exact test/development overlaps: {overlap.sum()}")


if __name__ == "__main__":
    main()
