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
- [.] Runner:
    - [X] Topological tree generation
    - [X] Async runner experiments
    - [ ] Implement async runner
    - [ ] Implement initial blocks:
        - [ ] Data sources:
            - [ ] Image
            - [ ] Video
            - [ ] RTSP stream
            - [ ] s3 bucket
        - [ ] Operators:
            - [ ] Image augmentations:
                - [ ] Resize
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

