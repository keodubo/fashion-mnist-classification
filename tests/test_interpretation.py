"""Numerical contracts for the offline CNN interpretation helpers."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "presentation/editorial"))
from build_interpretation import gradient_saliency, occlusion_sensitivity, occlusion_patches


class InterpretationTests(unittest.TestCase):
    def model(self, weights):
        layer = nn.Linear(4, 2)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([weights, [0., 0., 0., 0.]]))
            layer.bias.copy_(torch.tensor([1., 2.]))
        return nn.Sequential(nn.Flatten(), layer).eval()

    def test_saliency_is_absolute_derivative_of_the_selected_logit(self):
        model = self.model([1., -2., 3., -4.])
        image = torch.tensor([[[[.1, .2], [.3, .4]]]])
        np.testing.assert_allclose(gradient_saliency(model, image, 0), [[1, 2], [3, 4]])
        np.testing.assert_array_equal(gradient_saliency(model, image, 1), np.zeros((2, 2)))

    def test_occlusion_retains_signed_score_drop_and_averages_covering_patches(self):
        model = self.model([1., -2., 3., -4.])
        image = torch.tensor([[[[.1, .2], [.3, .4]]]])
        out = occlusion_sensitivity(model, image, 0, patch_size=1, stride=1, fill=.5)
        expected = np.array([[-.4, .6], [-.6, .4]])
        np.testing.assert_allclose(out['occlusion'], expected, atol=2e-7)
        np.testing.assert_allclose(out['patch_deltas'], expected.ravel(), atol=2e-7)
        self.assertAlmostEqual(out['base_score'], 0., places=6)
        overlap = occlusion_sensitivity(model, image, 0, patch_size=2, stride=1, fill=.5)
        np.testing.assert_allclose(overlap['occlusion'], np.zeros((2, 2)), atol=2e-7)

    def test_edge_patches_cover_every_pixel_without_exceeding_the_image(self):
        patches = occlusion_patches(28, 28, patch_size=5, stride=2)
        self.assertEqual(len(patches), 169)
        coverage = np.zeros((28, 28))
        for p in patches:
            r, c, h, w = (p[k] for k in ('row', 'col', 'height', 'width'))
            self.assertLessEqual(r+h, 28)
            self.assertLessEqual(c+w, 28)
            coverage[r:r+h, c:c+w] += 1
        self.assertTrue((coverage > 0).all())
        self.assertEqual(patches[-1], {'row': 23, 'col': 23, 'height': 5, 'width': 5})

    def test_explanations_preserve_input_and_model_predictions(self):
        model = self.model([1., -2., 3., -4.])
        image = torch.tensor([[[[.1, .2], [.3, .4]]]])
        original = image.clone()
        before = model(image).detach().clone()
        gradient_saliency(model, image, 0)
        occlusion_sensitivity(model, image, 0, patch_size=1, stride=1)
        torch.testing.assert_close(model(image), before, rtol=0, atol=0)
        torch.testing.assert_close(image, original, rtol=0, atol=0)

    def test_invalid_patch_configuration_and_nonfinite_input_are_rejected(self):
        with self.assertRaises(ValueError):
            occlusion_patches(2, 2, patch_size=3)
        with self.assertRaises(ValueError):
            occlusion_patches(28, 28, stride=0)
        image = torch.full((1, 1, 2, 2), float('nan'))
        with self.assertRaises(ValueError):
            gradient_saliency(self.model([1., 1., 1., 1.]), image, 0)


if __name__ == '__main__':
    unittest.main()
