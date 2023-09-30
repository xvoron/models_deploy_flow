import json
from server import BuildRequest


def get_test_data():
    with open("data2.json", "r") as f:
        data = json.load(f)
    return BuildRequest(**data)


