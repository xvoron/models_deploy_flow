import asyncio
import numpy as np

def io_trace(func):
    async def wrapper(self, input_data):
        print(f"Start {self.id}")
        self.input_data = input_data
        result = await func(self, input_data)
        self.output_data = result
        print(f"End {self.id}")
        return result
    return wrapper

class VideoSource:
    def __init__(self, config):
        self.config = config
        self.objects_to_iterate = [np.random.rand(5, 5, 3) for _ in range(2)]
        self.actual_position = 0

        self.end = False

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.objects_to_iterate) > self.actual_position:
            self.actual_position += 1
            return self.objects_to_iterate[self.actual_position - 1]

        self.end = True

class VideoSource2:
    def __init__(self, config):
        self.config = config
        self.objects_to_iterate = [np.random.rand(5, 5, 3) for _ in range(1)]
        self.actual_position = 0

        self.end = False

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.objects_to_iterate) > self.actual_position:
            self.actual_position += 1
            return self.objects_to_iterate[self.actual_position - 1]

        self.end = True

class VideoNode2:
    def __init__(self):
        self.id = 'video_node2'
        self.parents = []
        self.config = {}

        self.video_source = VideoSource(self.config)

    @io_trace
    async def run(self, input_data):
        await asyncio.sleep(1)
        return next(self.video_source)

class VideoNode:
    def __init__(self):
        self.id = 'video_node'
        self.parents = []
        self.config = {}

        self.video_source = VideoSource(self.config)

    @io_trace
    async def run(self, input_data):
        await asyncio.sleep(1)
        return next(self.video_source)

class ModelNode:
    def __init__(self):
        self.id = 'model_node'
        self.parents = ['video_node', 'video_node2']
        self.config = {}

    @io_trace
    async def run(self, input_data):
        await asyncio.sleep(1)
        return [np.random.rand(5, 1) for _ in range(len(input_data))]

class VisualizeNode:
    def __init__(self):
        self.id = 'visualize_node'
        self.parents = ['video_node', 'video_node2', 'model_node']
        self.config = {}

    @io_trace
    async def run(self, input_data):
        await asyncio.sleep(1)
        return input_data, "Visualized!!!"

async def execute_node(node, nodes_dict, cache):
    # Check if the result is already in the cache
    if node.id in cache:
        return cache[node.id]

    # Execute parent nodes and store their results in the cache
    input_data = await asyncio.gather(*[execute_node(nodes_dict[parent], nodes_dict, cache) for parent in node.parents])
    result = await node.run(input_data)

    # Store the result in the cache
    cache[node.id] = result

    return result

async def main():
    nodes = [VideoNode(), VideoNode2(), ModelNode(), VisualizeNode()]
    nodes_dict = {node.id: node for node in nodes}

    emit_nodes = ["visualize_node"]

    sources = [node for node in nodes if isinstance(node, VideoNode)]

    while True:
        # Check if all sources have ended
        if all([source.video_source.end for source in sources]):
            break

        # Create a cache dictionary to store results
        cache = {}

        for node in nodes:
            result = await execute_node(node, nodes_dict, cache)

        # Emit the results of the nodes that need to be emitted
        for node in nodes:
            if node.id in emit_nodes:
                print(f"{node.id} emitted: {node.output_data}")

if __name__ == '__main__':
    asyncio.run(main())
