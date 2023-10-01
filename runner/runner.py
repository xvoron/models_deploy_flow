from __future__ import annotations
import asyncio
from typing import Any, Optional

from pydantic import BaseModel, validator

from core.operators import OPS, Op, Video


def io_trace(func):
    def wrapper(self, x):
        print(f"Calling {self.op.name} with {x}")
        self.inputs = x
        result = func(self, x)
        self.outputs = result
        print(f"Returning {self.op.name} with {result}")
        return result
    return wrapper


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
    childrens: list[Node]= []
    parents: list[Node] = []
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
        self.op = OPS[label](config)

    @validator("input_name", "output_name")
    def validate_io(cls, v, values):
        """Validate if self.op contains a input_name and output_name"""
        if v is None:
            return values["op"].input_name if "input_name" else values["op"].output_name

    @io_trace
    async def run(self, x):
        assert self.op is not None
        return self.op(x)

def build_tree(nodes, edges):
    node_dict = {node.id: node for node in nodes}

    for edge in edges:
        source_node = node_dict.get(edge.source)
        target_node = node_dict.get(edge.target)

        if source_node and target_node:
            source_node.childrens.append(target_node.id)
            target_node.parents.append(source_node.id)

def validate_tree(tree):
    assert tree.parents is None


def print_tree(tree, level=0):
    if tree is None:
        return
    print("\t" * level + tree.id)
    if tree.children is not None:
        for child in tree.children:
            print_tree(child, level + 1)


async def execute_node(node, nodes_dict, cache):
    # Check if the result is already in the cache
    if node.id in cache:
        return cache[node.id]

    # Execute parent nodes and store their results in the cache
    input_data = await asyncio.gather(*[execute_node(nodes_dict[parent], nodes_dict, cache) for parent in node.parents])
    result = await node.run(input_data)

    # Store the result in the cache
    cache[node.id] = result


async def main(nodes):
    nodes_dict = {node.id: node for node in nodes}

    emit_nodes = ["4"]

    sources = [node for node in nodes if isinstance(node.op, Video)]

    while True:
        # Check if all sources have ended
        if all([source.op.video_source.end for source in sources]):
            break

        # Create a cache dictionary to store results
        cache = {}

        for node in nodes:
            result = await execute_node(node, nodes_dict, cache)

        # Emit the results of the nodes that need to be emitted
        for node in nodes:
            if node.id in emit_nodes:

                print(f"{node.id} emitted: {node.outputs}")

if __name__ == "__main__":
    from utils import get_test_data
    build_request = get_test_data()
    nodes = build_request.nodes
    edges = build_request.edges
    build_tree(nodes, edges)
    print(nodes)

    asyncio.run(main(nodes))
