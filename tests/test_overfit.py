import os
import random
import time
import unittest
from pathlib import Path


@unittest.skipUnless(os.getenv('RUN_SLOW') == '1', 'set RUN_SLOW=1 to run the CPU overfit smoke test')
class OverfitSmokeTest(unittest.TestCase):
    def test_yolov7_tiny_overfits_one_synthetic_image(self):
        import numpy as np
        import torch
        import yaml

        from models.yolo import Model
        from utils.general import box_iou, non_max_suppression
        from utils.loss import ComputeLossOTA

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        old_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_num_threads)

        repo_root = Path(__file__).resolve().parents[1]
        model = Model(str(repo_root / 'cfg/training/yolov7-tiny.yaml'), nc=1).cpu()
        with open(repo_root / 'data/hyp.scratch.tiny.yaml') as hyp_file:
            model.hyp = yaml.safe_load(hyp_file)
        model.gr = 1.0
        criterion = ComputeLossOTA(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        image = torch.ones((1, 3, 128, 128))
        image[:, 0, 44:84, 44:84] = 0.0
        image[:, 2, 44:84, 44:84] = 0.0
        targets = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.3125, 0.3125]])
        deadline = time.monotonic() + 600
        losses = []

        model.train()
        for _ in range(2000):
            optimizer.zero_grad()
            predictions = model(image)
            loss, _ = criterion(predictions, targets, image)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
            if time.monotonic() >= deadline:
                break

        first_losses = losses[:10]
        last_losses = losses[-50:]
        first_average = sum(first_losses) / len(first_losses) if first_losses else float('nan')
        last_average = sum(last_losses) / len(last_losses) if last_losses else float('nan')
        loss_ratio = last_average / first_average if first_average else float('inf')

        model.eval()
        with torch.no_grad():
            inference, _ = model(image)
        detections = non_max_suppression(inference, conf_thres=0.25, iou_thres=0.45)[0]
        ground_truth = torch.tensor([[44.0, 44.0, 84.0, 84.0]])
        if len(detections):
            highest_score = detections[detections[:, 4].argmax()]
            best_iou = box_iou(highest_score[:4].view(1, 4), ground_truth).item()
            best_box = highest_score[:4].tolist()
            best_score = highest_score[4].item()
        else:
            best_iou, best_box, best_score = 0.0, None, 0.0

        with self.subTest('loss convergence'):
            self.assertGreaterEqual(
                len(losses),
                60,
                f'CPU time limit reached after {len(losses)} steps; first-10 average='
                f'{first_average:.6f}, last-50 average={last_average:.6f}, '
                f'NMS detections={len(detections)}, highest-score IoU={best_iou:.4f}',
            )
            if len(losses) >= 60:
                self.assertLessEqual(
                    last_average,
                    first_average * 0.20,
                    f'loss threshold missed after {len(losses)} steps: first-10 average='
                    f'{first_average:.6f}, last-50 average={last_average:.6f}, '
                    f'ratio={loss_ratio:.4f}, NMS detections={len(detections)}, '
                    f'highest-score IoU={best_iou:.4f}',
                )

        with self.subTest('NMS detections'):
            self.assertGreater(
                len(detections),
                0,
                f'NMS returned no detections after {len(losses)} steps; '
                f'first-10 average={first_average:.6f}, last-50 average={last_average:.6f}',
            )

        with self.subTest('highest-score box IoU'):
            self.assertGreater(
                best_iou,
                0.5,
                f'highest-score detection IoU={best_iou:.4f}, expected > 0.5; '
                f'box={best_box}, score={best_score:.4f}, training steps={len(losses)}, '
                f'first-10 average={first_average:.6f}, last-50 average={last_average:.6f}',
            )


if __name__ == '__main__':
    unittest.main()
