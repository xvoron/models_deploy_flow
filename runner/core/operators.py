from __future__ import annotations
from abc import ABC, abstractmethod

import cv2
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel

from .basic_types import ImageType


class Op(ABC):
    def __init__(self, config):
        self.config = config
        self.input_types: list = []
        self.output_types: list = []


    def __call__(self, x):
        return x

    @abstractmethod
    def _name(self):
        raise NotImplementedError

    @property
    def name(self):
        return self._name()

    def __repr__(self):
        return f"{self.name}:{self.config}"

    def __str__(self):
        return self.__repr__()


class Image(Op):
    class Config(BaseModel):
        url: str

    def __init__(self, config: dict):
        super().__init__(config)
        self.config = self.Config(**config)
        self.input_types = [list, None]
        self.output_types = [ImageType]

    def __call__(self, x):
        image = ImageType(url=self.config.url)
        image.load().to_rgb()
        return image

    def _name(self):
        return "Image"

class VideoSource:
    def __init__(self, config):
        self.config = config
        self.objects_to_iterate = [ImageType(image=np.random.rand(224, 224, 3)) for _ in range(2)]
        self.actual_position = 0

        self.end = False

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.objects_to_iterate) > self.actual_position:
            self.actual_position += 1
            return self.objects_to_iterate[self.actual_position - 1]

        self.end = True

class Video(Op):
    class Config(BaseModel):
        url: str

    def __init__(self, config: dict):
        super().__init__(config)
        self.config = self.Config(**config)
        self.input_types = [list, None]
        self.output_types = [ImageType]

        self.video_source = VideoSource(config)

    def __call__(self, x):
        return next(self.video_source)

    def _name(self):
        return "Video"


class Resize(Op):
    class Config(BaseModel):
        size: list[int]

    def __init__(self, config: dict):
        super().__init__(config)
        self.config = self.Config(**config)
        self.input_types = [list[ImageType], ImageType]
        self.output_types = [list[ImageType], ImageType]

    def __call__(self, input_data: list):
        images = [x for x in input_data if isinstance(x, ImageType)]
        for image in images:
            if image.image is not None:
                image.image = cv2.resize(image.image, tuple(self.config.size))
        return images

    def _name(self):
        return "Resize"


class Model(Op):
    class Config(BaseModel):
        url: str

    def __init__(self, config: dict):
        super().__init__(config)
        self.config = self.Config(**config)

        self._session = self._load_model(config)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    @property
    def input_name(self):
        return self._input_name

    @property
    def output_name(self):
        return self._output_name

    @staticmethod
    def _load_model(config):
        return ort.InferenceSession(config["url"], providers=["CUDAExecutionProvider"])

    def __call__(self, input_data):
        # create_batch from images
        print(input_data)
        batch = []
        for x in input_data:
            if isinstance(x, ImageType):
                batch.append(x.image)

        if len(batch) == 0:
            print("No images in batch")
            return None

        x = np.array(batch)
        # (B, H, W, C) -> (B, C, H, W)
        x = np.transpose(x, (0, 3, 1, 2))
        x = x.astype(np.float32) / 255.0
        return self._session.run([self._output_name], {self._input_name: x})[0]

    def _name(self):
        return "Model"


class Visualizer(Op):
    def __call__(self, x):
        print(x)

    def _name(self):
        return "Visualizer"


class Output(Op):
    def __call__(self, x):
        return x

    def _name(self):
        return "Output"


class ClassificationPredictions(Op):
    def __call__(self, x):
        return x

    def _name(self):
        return "ClassificationPredictions"


OPS = {
    "Image": Image,
    "Video": Video,
    "Resize": Resize,
    "Model": Model,
    "Visualizer": Visualizer,
    "Output": Output,
    "ClassificationPredictions": ClassificationPredictions,
}
