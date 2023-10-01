from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np


@dataclass
class ImageType:
    url: Optional[str] = None
    image: Optional[np.ndarray] = None
    form: Literal["rgb", "bgr"] = "rgb"

    def load(self):
        assert self.url is not None
        self.image = cv2.imread(self.url)
        self.form = "bgr"
        return self

    def to_rgb(self):
        assert self.image is not None
        if not self.form == "rgb":
            self.form = "rgb"
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def to_bgr(self):
        assert self.image is not None
        if not self.form == "bgr":
            self.form = "bgr"
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

    @property
    def shape(self):
        if self.image is None:
            return None
        return self.image.shape

    def __repr__(self):
        return f"ImageType(url={self.url}, form={self.form}, shape={self.shape})"

    def __str__(self):
        return self.__repr__()


class ModelPredictions:
    ...
