# Models Deploy Flow Builder

## Overview
This is a simple tool to help you deploy your models to production using simple
UI.

## TODO:
- [.] Simple test UI:
    - [X] Flow builder
    - [ ] Custom blocks
    - [ ] Custom blocks with configuration
    - [ ] Save flow
    - [ ] Load flow
    - [ ] Run flow
    - [ ] Visualizations using sockets from the backend
- [o] Runner:
    - [X] Topological tree generation
    - [X] Async runner experiments
    - [X] Implement async runner
    - [o] Implement initial blocks:
        - [o] Data sources:
            - [X] Image
            - [X] Video
            - [ ] RTSP stream
            - [ ] s3 bucket
        - [.] Operators:
            - [.] Image augmentations:
                - [X] Resize
                - [ ] Crop
                - [ ] Rotate
                - [ ] etc.
            - [ ] Models:
                - [ ] Learn about ONNX standards
            - [ ] Post-processing:
                - [ ] Segmentation
                - [ ] ???
            - [ ] Visualizations:
                - [ ] Scope Block
                - [ ] Bounding boxes
                - [ ] Segmentation masks
                - [ ] ???
    - [ ] Server side
    - [ ] Sockets
- [ ] Tests:
    - [ ] Runner
    - [ ] UI

