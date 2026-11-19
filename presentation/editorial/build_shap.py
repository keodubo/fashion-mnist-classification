"""Exact grouped interventional Shapley values for the CNN Coat probability."""
from itertools import combinations
from math import comb
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from final_evaluation import FILES, build_cnn, read_idx, sha

REGIONS = ["Upper left", "Upper right", "Lower left", "Lower right"]
GRID = (np.arange(28)[:, None] // 14) * 2 + np.arange(28)[None, :] // 14
TARGET = 4


def stratified(ids, labels, count, rng):
    return np.concatenate([rng.choice(ids[labels[ids] == c], count, replace=False)
                           for c in range(10)])


def explain(model, image, background):
    values = []
    for mask in range(16):
        keep = (mask & (1 << GRID)) != 0
        hybrids = np.where(keep, image, background)
        inputs = torch.from_numpy(hybrids[:, None].astype(np.float32) / 255)
        values.append(float(model(inputs).softmax(1)[:, TARGET].double().mean()))
    phi = np.zeros(4)
    for region in range(4):
        others = sorted(set(range(4)) - {region})
        for size in range(4):
            for subset in combinations(others, size):
                mask = sum(1 << j for j in subset)
                phi[region] += (values[mask | (1 << region)] - values[mask]) / (4 * comb(3, size))
    np.testing.assert_allclose(values[0] + phi.sum(), values[-1], atol=1e-7)
    return values[0], values[-1], phi


def plot(evidence):
    phi = np.array(evidence["contributions"])
    brightness = np.array(evidence["mean_intensity"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    order = np.argsort(-np.abs(phi).mean(0))
    a.axvline(0, color="#999999", lw=1)
    for y, j in enumerate(order):
        bins = np.floor(phi[:, j] / .018).astype(int)
        for group in np.unique(bins):
            ids = np.flatnonzero(bins == group)
            ids = ids[np.argsort(phi[ids, j])]
            offsets = np.arange(len(ids)) - (len(ids) - 1) / 2
            dots = a.scatter(phi[ids, j], y + offsets * min(.08, .68 / len(ids)),
                             c=brightness[ids, j], cmap="coolwarm", norm=Normalize(0, 1),
                             s=13, alpha=.85, linewidths=0)
    a.set_yticks(range(4), [REGIONS[i] for i in order])
    a.invert_yaxis()
    a.set_title("100 validation images", loc="left", fontweight="bold")
    a.set_xlabel("SHAP value for Coat probability")
    fig.colorbar(dots, ax=a, orientation="horizontal", shrink=.62, pad=.18,
                 label="Mean region intensity (0–1)")
    local = np.array(evidence["local_contributions"])
    x = evidence["base"]
    order = np.argsort(-abs(local))
    for y, j in enumerate(order):
        delta = local[j]
        b.barh(y, delta, left=x, height=.55, color="#ac364b" if delta > 0 else "#326b9f")
        b.text(x + delta / 2, y - .34, f"{delta:+.3f}", ha="center", fontsize=10)
        x += delta
        if y < 3:
            b.plot([x,x], [y+.275,y+.725], ls=":", color=".45", lw=1)
    b.axvline(evidence["base"], ls=":", color=".5")
    b.set_yticks(range(4), [REGIONS[i] for i in order])
    b.set_ylim(3.7,-.75)
    b.set_xlabel("Predicted Coat probability")
    b.set_xlim(min(0, b.get_xlim()[0]), max(1, b.get_xlim()[1]))
    b.set_title(f"Shirt #32, predicted Coat\nBaseline {evidence['base']:.3f} → prediction {x:.3f}",
                loc="left", fontweight="bold")
    for ax in (a,b):
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.savefig(HERE / "coat-shap.png", dpi=160)
    plt.close(fig)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    rng = np.random.default_rng(42)
    name, fingerprint = FILES["CNN"]
    checkpoint = ROOT / "results" / name
    assert sha(checkpoint) == fingerprint
    model = build_cnn().eval()
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    split = np.load(ROOT / "data/splits/train-validation-v1.npz")
    images = read_idx(ROOT / "data/raw/train-images-idx3-ubyte.gz")
    labels = read_idx(ROOT / "data/raw/train-labels-idx1-ubyte.gz")
    bg = stratified(split["train_indices"], labels, 2, rng)
    ids = stratified(split["validation_indices"], labels, 10, rng)
    demo = next(x for x in json.loads((HERE / "data.json").read_text())["demos"] if x["id"] == 32)
    assert 32 in split["validation_indices"] and labels[32] == 6
    base, pred, local = explain(model, images[32], images[bg])
    prob = demo["cnn_probabilities"]
    assert np.argmax(prob) == TARGET
    np.testing.assert_allclose(pred, prob[TARGET], atol=1e-6)
    results = [explain(model, images[i], images[bg])[2] for i in ids]
    brightness = [[images[i][GRID == j].mean()/255 for j in range(4)] for i in ids]
    evidence = dict(target="Coat probability", checkpoint_sha256=fingerprint, regions=REGIONS,
                    background_training_ids=bg.tolist(), validation_ids=ids.tolist(),
                    contributions=np.array(results).tolist(), mean_intensity=brightness,
                    base=base, local_id=32, prediction=pred, local_contributions=local.tolist())
    (HERE / "coat-shap.json").write_text(json.dumps(evidence, indent=2) + "\n")
    plot(evidence)
    print(base, pred, local, np.abs(results).mean(0))


if __name__ == "__main__":
    main()
