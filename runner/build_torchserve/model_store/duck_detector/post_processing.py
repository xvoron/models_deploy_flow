from typing import List, Tuple, Dict

import numpy as np
import torch

from aiis_core import Box


def get_boxes_from_torchvision(torchvision_boxes: Dict, classes: List[str],
                               shape: Tuple[int, int], threshold: float = 0.) -> List[Box]:
    h, w = shape

    boxes = torchvision_boxes['boxes'].detach().cpu().numpy()
    labels = torchvision_boxes['labels'].detach().cpu().numpy()
    scores = torchvision_boxes.get('scores', torch.ones(len(labels))).detach().cpu().numpy()

    aiis_boxes = []
    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box / np.array([w, h, w, h])
        label = classes[label - 1]
        score = float(score.item())
        if score >= threshold:
            aiis_boxes.append(Box.from_xyxy(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                            classification={label: score}, is_normalized=True))
    return aiis_boxes
