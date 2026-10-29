import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from utils.loss import ComputeLossAuxOTA, ComputeLossBinOTA, ComputeLossOTA


class DummyModel(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.dummy_parameter = nn.Parameter(torch.zeros(1))
        self.gr = 1.0
        self.hyp = {
            'anchor_t': 4.0,
            'box': 1.0,
            'cls': 1.0,
            'cls_pw': 1.0,
            'fl_gamma': 0.0,
            'label_smoothing': 0.0,
            'obj': 1.0,
            'obj_pw': 1.0,
        }
        detect = SimpleNamespace(
            anchors=torch.tensor([[[1.0, 1.0]]]),
            na=1,
            nc=1,
            nl=1,
            stride=torch.tensor([8]),
        )
        if loss_type is ComputeLossBinOTA:
            detect.bin_count = 2
        self.model = [detect]


class ComputeLossOTATest(unittest.TestCase):
    losses = (ComputeLossOTA, ComputeLossBinOTA, ComputeLossAuxOTA)

    def make_predictions(self, loss_type, requires_grad=False):
        if loss_type is ComputeLossBinOTA:
            channels = 2 * (2 + 1) + 2 + 1 + 1
        else:
            channels = 6
        levels = 2 if loss_type is ComputeLossAuxOTA else 1
        return [
            torch.zeros((1, 1, 4, 4, channels), requires_grad=requires_grad)
            for _ in range(levels)
        ]

    def test_empty_targets_match_objectness_reference(self):
        for loss_type in self.losses:
            with self.subTest(loss=loss_type.__name__):
                criterion = loss_type(DummyModel(loss_type))
                predictions = self.make_predictions(loss_type)
                targets = torch.zeros((0, 6))
                images = torch.zeros((1, 3, 32, 32))

                loss, items = criterion(predictions, targets, images)

                objectness_scale = 5.0 if loss_type is ComputeLossAuxOTA else 4.0
                expected_objectness = torch.tensor(objectness_scale * torch.log(torch.tensor(2.0)))
                torch.testing.assert_close(
                    items,
                    torch.stack((torch.tensor(0.0), expected_objectness, torch.tensor(0.0), expected_objectness)),
                )
                torch.testing.assert_close(loss, expected_objectness)

    def test_positive_targets_produce_finite_loss_and_gradients(self):
        for loss_type in self.losses:
            with self.subTest(loss=loss_type.__name__):
                criterion = loss_type(DummyModel(loss_type))
                predictions = self.make_predictions(loss_type, requires_grad=True)
                targets = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.25, 0.25]])
                images = torch.zeros((1, 3, 32, 32))

                loss, items = criterion(predictions, targets, images)

                self.assertTrue(torch.isfinite(loss).all())
                self.assertTrue(torch.isfinite(items).all())
                loss.backward()
                for prediction in predictions:
                    self.assertIsNotNone(prediction.grad)
                    self.assertTrue(torch.isfinite(prediction.grad).all())
                    self.assertGreater(prediction.grad.abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
