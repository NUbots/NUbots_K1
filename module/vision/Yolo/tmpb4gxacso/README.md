# Yolo

## Description

This module integrates a YOLO (You Only Look Once) model to identify and classify objects within images. The default model is Booster Robotics' RoboCup demo detector (`booster.onnx`). Its classes (matching the `classnames` list in robocup_demo's `src/vision/config/vision.yaml`) are:

- `Ball`
- `Goalpost`
- `Person` (treated as a robot detection)
- `LCross`, `TCross`, `XCross` (field line intersections)
- `PenaltyPoint`
- `Opponent` (treated as a robot detection)
- `BRMarker` (field marker)

Confidence thresholds for each class can be specified in the config.

Penalty point and BRMarker detections have no dedicated message type yet, so they are only emitted as a `message::vision::BoundingBox` for visualisation/debugging in NUsight.

Inference can be ran on either the CPU or GPU using OpenVino (https://github.com/openvinotoolkit/openvino).

## Usage

Include this module to detect balls, goals, robots, field line intersections, penalty points and field markers in images.

If the GreenHorizon is included in the program, balls, field line intersections and robots outside of the GreenHorizon will be discarded. Penalty point and marker bounding boxes are not filtered by the GreenHorizon.

To run with GPU device in docker you need to include the following flags `./b run {binary} --gpus all`

## Consumes

- `message::input::Image` the image to run the YOLO on.

## Emits

- `message::vision::Balls` ball detections
- `message::vision::Goals` goal detections
- `message::vision::Robots` robot detections (from both "Person" and "Opponent" classes)
- `message::vision::FieldIntersections` field line intersections
- `message::vision::BoundingBoxes` bounding boxes for every detected class, including penalty points and markers

## Dependencies

- [OpenVino](https://github.com/openvinotoolkit/openvino)
- [Eigen Linear Algebra Library](https://eigen.tuxfamily.org/index.php)
- [OpenCV](https://opencv.org/)
