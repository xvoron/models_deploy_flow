from __future__ import annotations
from server import BuildRequest
import json
from dataclasses import dataclass
from pydantic import BaseModel, Field
import yaml

@dataclass
class Op:
    ...

class Image(Op):
    image: str

    def __repr__(self):
        return 'input'

    def __str__(self):
        return 'input'

class Resize(Op):
    size: tuple[int, int]

    def __repr__(self):
        return 'resize'

    def __str__(self):
        return 'resize'

class Model(Op):
    model: str

    def __repr__(self):
        return 'model'

    def __str__(self):
        return 'model1'

class PostProcessing(Op):
    post_processing: str

    def __repr__(self):
        return 'post_processing'

    def __str__(self):
        return 'output'


def get_test_data():

    with open("data.json", "r") as f:
        data = json.load(f)
    return BuildRequest(**data)


def create_workflow(models: list[ModelMar], pairs: dict) -> Workflow:

    def create_models(models) -> Models:
        ret = Models()
        for model in models:
            ret.__setattr__(model.name, {'url': model.url})
        return ret

    def create_dag(pairs):
        dag = DAG()
        for parent, children in pairs.items():
            dag.__setattr__(parent, children)
        return dag

    return Workflow(models=create_models(models), dag=create_dag(pairs))


class Models(BaseModel):
    min_workers: int = Field(1, alias='min-workers')
    max_workers: int = Field(1, alias='max-workers')
    batch_size: int = Field(1, alias='batch-size')
    max_batch_delay: int = Field(0, alias='max-batch-delay')
    class Config:
        extra = 'allow'


class DAG(BaseModel):
    class Config:
        extra = 'allow'


class Workflow(BaseModel):
    models: Models
    dag: DAG


class ModelMar(BaseModel):
    name: str = 'duck_detector'
    url: str = 'duck_detector.mar'

class Config(BaseModel):
    model: ModelMar = ModelMar()
    size: tuple[int, int] = (100, 100)
    
def convert_names(label: str):
    match label:
        case "Image":
            return "pre_processing"
        case "Resize":
            return "resize"
        case "Model":
            return "duck_detector"
        case "Output":
            return "post_processing"


if __name__ == "__main__":
    build_request = get_test_data()
    nodes = build_request.nodes
    edges = build_request.edges

    # Create a dictionary to store parent/child pairs
    parent_child_pairs = {}

    # Iterate through the edges
    for edge in edges:
        source_id = edge.source
        target_id = edge.target

        # Find the corresponding nodes
        source_node = next(node for node in nodes if node.id == source_id)
        target_node = next(node for node in nodes if node.id == target_id)

         # Check if the parent already has children in the dictionary
        if source_node.data.label not in parent_child_pairs:
            parent_child_pairs[convert_names(source_node.data.label)] = []

        # Add the child to the list of children for the parent
        parent_child_pairs[convert_names(source_node.data.label)].append(convert_names(target_node.data.label))

    workflow = create_workflow([ModelMar()], parent_child_pairs)

    print(yaml.dump(workflow.model_dump(by_alias=True), sort_keys=False))
    with open('workflow.yaml', 'w') as f:
        yaml.dump(workflow.model_dump(by_alias=True), f, sort_keys=False)

