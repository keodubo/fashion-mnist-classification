"""Export gradient and occlusion evidence from the frozen CNN on validation data.

Run from the repository root:
    .venv/bin/python presentation/editorial/build_interpretation.py

No training, checkpoint writes, test loading or test-based tuning occurs. The
browser displays these stored measurements; it does not run a model.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def _image(image):
    if image.ndim != 4 or image.shape[:2] != (1, 1):
        raise ValueError("Expected one grayscale image with shape (1, 1, H, W)")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("Image must contain finite floating-point values")
    return image.detach().clone()


def _score(outputs, target):
    if outputs.ndim != 2 or target < 0 or target >= outputs.shape[1]:
        raise ValueError("Target must identify an output class")
    if not torch.isfinite(outputs).all():
        raise ValueError("Model produced nonfinite class scores")
    return outputs[:, target]


def gradient_saliency(model, image, target):
    """Return |d selected raw logit / d pixel| as an H-by-W NumPy array.

    The model must already be in evaluation mode. autograd.grad requests only
    input derivatives; it does not accumulate parameter gradients or update
    weights. A class score independent of the image has zero saliency.
    """
    if model.training:
        raise ValueError("Use model.eval() before computing explanations")
    with torch.enable_grad():
        x = _image(image).requires_grad_(True)
        score = _score(model(x), target).sum()
        gradient = (torch.autograd.grad(score, x, allow_unused=True)[0]
                    if score.requires_grad else None)
    if gradient is None:
        return np.zeros(tuple(x.shape[-2:]), dtype=np.float32)
    values = gradient.detach().abs()[0, 0].cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite input derivative")
    return values


def occlusion_patches(height, width, patch_size=5, stride=2):
    """Return full, unpadded patches, adding the final edge position as needed."""
    if (not all(isinstance(v, int) for v in (height, width, patch_size, stride))
            or min(height, width, patch_size, stride) < 1
            or patch_size > min(height, width) or stride > patch_size):
        raise ValueError("Use positive dimensions and 1 <= stride <= patch size <= image")

    def positions(size):
        starts = list(range(0, size - patch_size + 1, stride))
        if starts[-1] != size - patch_size:
            starts.append(size - patch_size)
        return starts

    return [{"row": row, "col": col, "height": patch_size, "width": patch_size}
            for row in positions(height) for col in positions(width)]


def occlusion_sensitivity(model, image, target, patch_size=5, stride=2, fill=.5, batch_size=64):
    """Return signed original-minus-occluded logits and their overlap mean map.

    The target class stays fixed even when an occluded image's winning class
    changes. Values below zero mean occlusion increased that target's score.
    """
    if model.training:
        raise ValueError("Use model.eval() before computing explanations")
    if not np.isfinite(fill) or not 0 <= fill <= 1 or batch_size < 1:
        raise ValueError("Use a finite fill in [0,1] and positive batch size")
    x = _image(image)
    height, width = x.shape[-2:]
    patches = occlusion_patches(height, width, patch_size, stride)
    with torch.no_grad():
        baseline = float(_score(model(x), target).item())
        deltas = []
        for start in range(0, len(patches), batch_size):
            chunk = patches[start:start + batch_size]
            batch = x.repeat(len(chunk), 1, 1, 1)
            for i, p in enumerate(chunk):
                row, col = p['row'], p['col']
                batch[i, :, row:row + patch_size, col:col + patch_size] = fill
            deltas.extend((baseline - _score(model(batch), target)).cpu().tolist())
    values = np.zeros((height, width), dtype=np.float64)
    coverage = np.zeros_like(values)
    for p, delta in zip(patches, deltas):
        region = np.s_[p['row']:p['row'] + patch_size, p['col']:p['col'] + patch_size]
        values[region] += delta
        coverage[region] += 1
    if not (coverage > 0).all() or not np.isfinite(deltas).all():
        raise ValueError("Occlusion must cover every pixel with finite measurements")
    return {"base_score": baseline, "patches": patches,
            "patch_deltas": deltas, "occlusion": values / coverage}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    from final_evaluation import FILES, build_cnn, read_idx

    torch.set_num_threads(4)
    source_path = HERE / "data.json"
    source = json.loads(source_path.read_text())
    checkpoint_relative, expected_hash = FILES['CNN']
    checkpoint = ROOT / "results" / checkpoint_relative
    if _sha(checkpoint) != expected_hash:
        raise RuntimeError("Frozen checkpoint fingerprint mismatch")
    model = build_cnn()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    model.requires_grad_(False)
    state_before = _state_sha(model)
    images_path = ROOT / "data/raw/train-images-idx3-ubyte.gz"
    labels_path = ROOT / "data/raw/train-labels-idx1-ubyte.gz"
    split_path = ROOT / "data/splits/train-validation-v1.npz"
    images, labels = read_idx(images_path), read_idx(labels_path)
    with np.load(split_path) as split:
        validation_ids = split['validation_indices']
    examples = []
    for demo in source['demos']:
        index, actual = int(demo['id']), int(demo['label'])
        expected_id = int(validation_ids[labels[validation_ids] == actual][0])
        if index != expected_id or labels[index] != actual:
            raise RuntimeError("Demo no longer matches the deterministic validation policy")
        np.testing.assert_array_equal(images[index].ravel(), demo['pixels'])
        x = torch.from_numpy(images[index].copy().astype(np.float32) / 255).reshape(1, 1, 28, 28)
        with torch.no_grad():
            logits = model(x)[0]
            probabilities = logits.softmax(0).numpy()
        np.testing.assert_allclose(probabilities, demo['cnn_probabilities'], rtol=2e-5, atol=2e-6)
        predicted = int(logits.argmax())
        target_ids = list(dict.fromkeys([predicted, actual]))
        targets = []
        for target in target_ids:
            gradient = gradient_saliency(model, x, target)
            occlusion = occlusion_sensitivity(model, x, target)
            targets.append({"class_id": target, "base_score": occlusion['base_score'],
                            "saliency": gradient.ravel().tolist(),
                            "occlusion": occlusion['occlusion'].ravel().tolist(),
                            "patch_deltas": occlusion['patch_deltas']})
        examples.append({"id": index, "label": actual, "predicted": predicted,
                         "logits": logits.tolist(), "targets": targets})
        print(f"Validation #{index}: {len(targets)} target(s), "
              f"saliency max={max(targets[0]['saliency']):.5f}, "
              f"occlusion range=[{min(targets[0]['occlusion']):.5f}, {max(targets[0]['occlusion']):.5f}]")
    if _sha(checkpoint) != expected_hash or _state_sha(model) != state_before:
        raise RuntimeError("Model state changed while generating explanations")
    output = {
        "checkpoint_sha256": expected_hash,
        "methods": {"score": "raw class logit", "gradient": "absolute input derivative",
                    "patch_size": 5, "stride": 2, "fill": .5,
                    "aggregation": "mean signed score drop over covering patches",
                    "display": "positive map values normalised independently to their maximum; signed patch deltas retained",
                    "target_policy": "predicted class, plus actual class when different; fixed within an explanation",
                    "occlusion_edge_policy": "include final full patch position to cover every pixel",
                    "caveat": "local model diagnostics, not causal evidence; patch fill can create out-of-distribution inputs"},
        "patches": occlusion_patches(28, 28),
        "examples": examples,
        "provenance": {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "input_split": "validation only; first saved validation source index per class",
            "training_images_sha256": _sha(images_path),
            "training_labels_sha256": _sha(labels_path),
            "split_sha256": _sha(split_path),
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": _state_sha(model),
            "script_sha256": _sha(Path(__file__)),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": "cpu", "computational_threads": 4,
            "test_data_loaded": False, "training_performed": False,
            "reference": "W10/10-CNNs-block2.pdf, pages 57-60",
        },
    }
    destination = HERE / "interpretation.json"
    destination.write_text(json.dumps(output, separators=(',', ':'), allow_nan=False) + '\n')
    print(f"Exported {len(examples)} validation examples to {destination.relative_to(ROOT)} ({destination.stat().st_size:,} bytes)")


if __name__ == '__main__':
    main()
