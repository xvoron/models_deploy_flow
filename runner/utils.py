import json
from server import BuildRequest


def get_test_data():
    with open("data.json", "r") as f:
        data = json.load(f)
    return BuildRequest(**data)


