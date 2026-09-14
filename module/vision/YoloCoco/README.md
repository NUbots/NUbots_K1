# YoloCoco

## Description

This module integrates a YOLO (You Only Look Once) model to identify and classify objects within images.

The classes for the model are from COCO (https://docs.ultralytics.com/datasets/detect/coco/#dataset-structure)

Confidence thresholds for each class can be specified in the config.

Inference is run on CPU using ONNX Runtime (https://onnxruntime.ai/).

## Usage

Include this module to detect balls, goals, robots and field line intersections in images.

## Consumes

- `message::input::Image` the image to run the YOLO on.

## Emits

- `message::vision::BoundingBoxes` bounding boxes of the detections

## Dependencies

- [ONNX Runtime](https://onnxruntime.ai/)
- [Eigen Linear Algebra Library](https://eigen.tuxfamily.org/index.php)
- [OpenCV](https://opencv.org/)
