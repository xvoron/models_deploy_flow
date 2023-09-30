from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any, Optional

import cv2
import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, validator

DEBUG = os.environ.get("DEBUG", False)

@dataclass
class Config:
    model_path: str


def io_tracker(func):
    def wrapper(self, x):
        if DEBUG:
            print(f"Calling {self.op.name} with {x}")
        self.inputs = x
        self.outputs = func(self, x)
        return self.outputs
    return wrapper

class Op(ABC):
    def __init__(self, config):
        self.config = config

    def __call__(self, x):
        return x

    @abstractmethod
    def _name(self):
        raise NotImplementedError

    @property
    def name(self):
        pass

    def __repr__(self):
        return f"{self._name()}:{self.config}"

    def __str__(self):
        return self.__repr__()


class Image(Op):
    class Config(BaseModel):
        url: str

    def __init__(self, config: dict):
        super().__init__(config)
        self.config = self.Config(**config)

    def __call__(self, x):
        assert x is None
        image = cv2.imread(self.config.url)     # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image    # RGB

    def _name(self):
        return "Image"


class Video(Op):
    class Config(BaseModel):
        url: str

    def __init__(self, config: dict):
        super().__init__(config)
        self.config = self.Config(**config)

    def __call__(self, x):
        assert x is None
        #cap = cv2.VideoCapture(self.config.url)
        #while cap.isOpened():
        #    ret, frame = cap.read()
        #    if ret:
        #        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #    else:
        #        break
        #    yield frame
        return np.random.randint(0, 255, (224, 224, 3))

    def _name(self):
        return "Video"


class Visualizer(Op):
    def __call__(self, x):
        print(x)
        ...

    def _name(self):
        return "Visualizer"





class Resize(Op):
    class Config(BaseModel):
        size: list[int]

    def __init__(self, config: dict):
        super().__init__(config)
        self.config = self.Config(**config)

    def __call__(self, x: np.ndarray):
        return cv2.resize(x, tuple(self.config.size))   # RGB

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
        return ort.InferenceSession(config["url"],
                                    providers=["CUDAExecutionProvider"])


    def __call__(self, x: np.ndarray):
        print(x.shape)
        # Add batch dimension
        x = np.expand_dims(x, axis=0) # RGB
        # (B, H, W, C) -> (B, C, H, W)
        x = np.transpose(x, (0, 3, 1, 2))
        x = x.astype(np.float32) / 255.0
        return self._session.run([self._output_name], {self._input_name: x})[0]

    def _name(self):
        return "Model"


class Output(Op):
    def __call__(self, x):
        return x

    def _name(self):
        return "Output"


class NodeData(BaseModel):
    label: str
    config: Optional[dict] = None


class Edge(BaseModel):
    id: str
    source: str
    target: str


class Node(BaseModel):
    id: str
    position: dict
    data: NodeData
    children: Optional[list[Node]] = None
    parent: Optional[list[Node]] = None
    inputs: Any = None
    outputs: Any = None
    op: Optional[Op] = None

    input_name: Optional[str] = None
    output_name: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        data = kwargs["data"]
        label = data.get("label")
        config = data.get("config", None)
        self.op = globals()[label](config)

    @validator("input_name", "output_name")
    def validate_io(cls, v, values):
        """Validate if self.op contains a input_name and output_name"""
        if v is None:
            return values["op"].input_name if "input_name" else values["op"].output_name

    @io_tracker
    def forward(self, x):
        assert self.op is not None
        return self.op(x)

def build_tree(nodes, edges):
    node_dict = {node.id: node for node in nodes}
    tree = None

    for edge in edges:
        source_node = node_dict.get(edge.source)
        target_node = node_dict.get(edge.target)

        if source_node and target_node:
            if source_node.children is None:
                source_node.children = []

            source_node.children.append(target_node)

            if target_node.parent is None:
                target_node.parent = []

            target_node.parent.append(source_node)
    for node in nodes:
        if node.parent is None:
            tree = node
    return tree

def validate_tree(tree):
    assert tree.parent is None


def forwardpass(node: Node, data) -> None:
    if node.children is None:
        node.forward(data)
    else:
        for child in node.children:
            forwardpass(child, node.forward(data))

def print_tree(tree, level=0):
    if tree is None:
        return
    print("\t" * level + tree.id)
    if tree.children is not None:
        for child in tree.children:
            print_tree(child, level + 1)


if __name__ == "__main__":
    from utils import get_test_data

    ids_to_return = ["4"]

    build_request = get_test_data()
    nodes = build_request.nodes
    edges = build_request.edges
    nodes_dict = {node.id: node for node in nodes}

    tree = build_tree(nodes, edges)
    print_tree(tree)
    validate_tree(tree)
    assert tree is not None

    forwardpass(tree, None)
    print([(nodes_dict[id].inputs, nodes_dict[id].outputs) for id in ids_to_return])

