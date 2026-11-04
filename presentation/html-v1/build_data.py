"""Export real dataset observations and frozen-model validation walkthroughs.

No fitting, hyperparameter selection, or final test inference is performed.
Run from the project root: .venv/bin/python presentation/html-v1/build_data.py
"""
from pathlib import Path
import base64
import hashlib
import io
import json
import sys

import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from final_evaluation import read_idx, build_cnn, CLASSES, FILES

def png(values):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

def heat(values):
    values = np.asarray(values)
    bound = max(float(np.abs(values).max()), 1e-12)
    pos = np.maximum(values, 0) / bound
    neg = np.maximum(-values, 0) / bound
    rgb = np.stack([20 + pos * 195 + neg * 230, 25 + pos * 225 + neg * 40, 23 + pos * 65 + neg * 120], axis=-1)
    return png(rgb.clip(0, 255))

def main():
    images = read_idx(ROOT / "data/raw/train-images-idx3-ubyte.gz")
    labels = read_idx(ROOT / "data/raw/train-labels-idx1-ubyte.gz")
    with np.load(ROOT / "data/splits/train-validation-v1.npz") as split:
        train_ids, val_ids = split["train_indices"], split["validation_indices"]
    saved = pd.read_csv(ROOT / "results/validation-comparison/aligned-validation-predictions.csv").set_index("source_index")
    models = {}
    for name, (relative, expected) in FILES.items():
        path = ROOT / "results" / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        if name == "CNN":
            model = build_cnn()
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            model.eval()
        else:
            model = joblib.load(path)
        models[name] = model
    torch.set_num_threads(4)
    corpus = []
    for category in range(10):
        for index in val_ids[labels[val_ids] == category][:8]:
            corpus.append({"id": int(index), "label": category, "image": png(images[index])})
    # One deterministic validation observation per class, without choosing by correctness.
    demos = []
    for category in range(10):
        index = int(val_ids[labels[val_ids] == category][0])
        x = images[index].reshape(1, -1).astype(np.float64) / 255
        with threadpool_limits(limits=4):
            distances, neighbour_rows = models["kNN"].kneighbors(x, n_neighbors=3)
            knn_probs = models["kNN"].predict_proba(x)[0]
            lr_scores = models["Logistic regression"].decision_function(x)[0]
            lr_probs = models["Logistic regression"].predict_proba(x)[0]
        neighbours = []
        for distance, row in zip(distances[0], neighbour_rows[0]):
            source_id = int(train_ids[row])
            neighbours.append({"id": source_id, "label": int(labels[source_id]), "distance": float(distance), "image": png(images[source_id])})
        predicted = int(lr_probs.argmax())
        contributions = models["Logistic regression"].coef_[predicted] * x[0]
        bias = float(models["Logistic regression"].intercept_[predicted])
        assert abs(contributions.sum() + bias - lr_scores[predicted]) < 1e-8
        tensor = torch.from_numpy(images[index].astype(np.float32) / 255).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            a = tensor
            layers = {}
            for layer_id, layer in enumerate(models["CNN"]):
                a = layer(a)
                if layer_id in (1, 2, 4, 5):
                    channels = a[0].numpy()
                    # A common intensity scale within each tensor preserves between-channel contrast.
                    mx = max(float(channels.max()), 1e-9)
                    layers[str(layer_id)] = [png(np.clip(ch / mx * 255, 0, 255)) for ch in channels[:8]]
            cnn_logits = a[0].numpy()
            cnn_probs = torch.softmax(a, dim=1)[0].numpy()
        for name, probs in [("Logistic regression", lr_probs), ("kNN", knn_probs), ("CNN", cnn_probs)]:
            assert int(probs.argmax()) == int(saved.loc[index, name]), (index, name)
        demos.append({"id": index, "label": category, "image": png(images[index]),
                      "pixels": images[index].ravel().tolist(), "neighbours": neighbours,
                      "knn_probabilities": knn_probs.tolist(), "lr_logits": lr_scores.tolist(),
                      "lr_probabilities": lr_probs.tolist(), "lr_bias": bias,
                      "lr_contribution_sum": float(contributions.sum()),
                      "lr_contribution_image": heat(contributions.reshape(28, 28)),
                      "lr_contribution_class": predicted, "cnn_logits": cnn_logits.tolist(),
                      "cnn_probabilities": cnn_probs.tolist(), "feature_maps": layers})
    final = ROOT / "results/final-evaluation-v1"
    test_images = read_idx(ROOT / "data/raw/t10k-images-idx3-ubyte.gz")
    errors = pd.read_csv(final / "sampled-errors.csv")
    sample_errors = []
    for _, row in errors.iterrows():
        sample_errors.append({"id": int(row.test_index), "model": row.model,
                              "actual": row.actual, "predicted": row.predicted,
                              "image": png(test_images[int(row.test_index)])})
    data = {"classes": CLASSES, "corpus": corpus, "demos": demos, "errors": sample_errors,
            "metrics": pd.read_csv(final / "metrics.csv").to_dict("records"),
            "per_class": pd.read_csv(final / "per-class-metrics.csv").to_dict("records"),
            "confusions": pd.read_csv(final / "confusions.csv").to_dict("records"),
            "cnn_matrix": pd.read_csv(final / "cnn-confusion-counts.csv", index_col=0).values.tolist(),
            "history": pd.read_csv(ROOT / "results/cnn-improvements/wider-history.csv").to_dict("records"),
            "logistic": pd.read_csv(ROOT / "results/logistic-regression-regularization/comparison.csv").to_dict("records"),
            "knn": pd.read_csv(ROOT / "results/knn-comparison/comparison.csv").to_dict("records"),
            "cnn": pd.read_csv(ROOT / "results/cnn-improvements/comparison.csv").to_dict("records"),
            "model_hashes": {n: sha for n, (_, sha) in FILES.items()},
            "notes": json.loads((HERE / "speaker-notes.json").read_text()),
            "provenance": {"walkthrough_split": "Validation", "examples": "First validation observation per class; no selection by correctness",
                           "features": "First 8 channels of each saved activation tensor; grayscale uses a common maximum within each tensor",
                           "attribution": "Logistic-regression image shows signed pixel × weight contributions for its predicted class, excluding the intercept; not a causal explanation",
                           "test": "Saved final-evaluation outputs only; no test inference performed to build this presentation"}}
    (HERE / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    manifest = {"demo_count": len(demos), "dataset_images": len(corpus), "prediction_parity": "All ten validation demos match saved predictions for all three models", "model_sha256": data["model_hashes"], "notes": data["provenance"]}
    (HERE / "evidence-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(corpus)} images, {len(demos)} real model walkthroughs, {len(sample_errors)} saved test errors. No training or test inference.")

if __name__ == "__main__":
    main()
