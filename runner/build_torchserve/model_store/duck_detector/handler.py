from __future__ import annotations
import json
from typing import Dict, List, Tuple

from aiis_core.handler_utils import (
    ModelOutput,
    Settings,
    encode_image,
    get_images,
    get_settings,
    get_transformations,
)
from aiis_core.visualizer import draw_boxes
import numpy as np
import torch
import torchvision  # noqa: F401 pylint: disable=unused-import
from ts.torch_handler.base_handler import BaseHandler

try:
    from post_processing import get_boxes_from_torchvision

except ImportError:
    from src.object_detection.utils.post_processing import get_boxes_from_torchvision


class ObjectDetectionHandler(BaseHandler):

    class ODSettings(Settings):
        score_threshold: float = 0.5

    def __init__(self):
        super().__init__()

        with open("config.json", "r") as fp:
            self._config = json.load(fp)

        self.w = self._config["shape"]["width"]
        self.h = self._config["shape"]["height"]
        self.classes = self._config["classes"]["object_detection"]

        self.transform = get_transformations('transforms.json')

    def preprocess(self, data: List[Dict]
                   ) -> Tuple[torch.Tensor, List[np.ndarray], List[ObjectDetectionHandler.ODSettings]]:
        images = get_images(data)
        settings = get_settings(data, settings_class=ObjectDetectionHandler.ODSettings)
        transformed_images = [self.transform(image=image)['image'] for image in images]

        return torch.stack(transformed_images), images, settings

    @torch.no_grad()
    def inference(self,
                  data: Tuple[torch.Tensor, List[np.ndarray], List[ObjectDetectionHandler.ODSettings]],
                  *args, **kwargs
                  ) -> Tuple[List[Dict[str, torch.Tensor]], List[np.ndarray], List[ObjectDetectionHandler.ODSettings]]:
        transformed_images, original_images, settings = data

        with torch.jit.optimized_execution(False):
            assert self.model is not None, "Model is not loaded"
            predictions = self.model([transformed_images.to(self.device).squeeze(), ], *args, **kwargs)[1]
            return predictions, original_images, settings

    def postprocess(self,
                    data: Tuple[List[Dict[str, torch.Tensor]], List[np.ndarray], List[ObjectDetectionHandler.ODSettings]]
                    ) -> List[Dict]:
        predictions, original_images, settings = data
        ret = []
        for prediction, image, settings in zip(predictions, original_images, settings):
            batch_results = {}
            boxes = get_boxes_from_torchvision(
                torchvision_boxes=prediction,
                classes=self.classes,
                shape=(self.h, self.w),
                threshold=settings.score_threshold)
            batch_results["boxes"] = boxes

            if settings.visualization:
                draw_boxes(image, boxes)
                batch_results["visualization"] = encode_image(image, 'jpg')

            ret.append(ModelOutput(**batch_results).dict(exclude_none=True))

        return ret
