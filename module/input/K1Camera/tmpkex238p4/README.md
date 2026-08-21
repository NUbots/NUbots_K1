# K1Camera

## Description

Subscribes directly to the Booster K1's camera topics using the [Booster Robotics
SDK](https://github.com/BoosterRobotics/booster_robotics_sdk) (1.7) over DDS, and emits the frames as
`message::input::Image`.

This module previously read shared memory segments written by
[NUbridge](https://github.com/NUbots/NUbridge). It no longer does, so no ROS 2 install or bridge
process is needed for imagery.

Images arriving as `nv12` (the K1 head camera's native format) are converted once to `RGB3`, the
codebase's FOURCC for packed 24-bit RGB. All other encodings are passed through with their matching
FOURCC code.

Each image is stamped with the capture time from the DDS message header rather than the time it was
received, and is matched against the `Sensors::Hcw` buffered closest to that time. If the capture
timestamp is more than a second away from our own clock it is distrusted, a warning is logged once,
and the receive time is used instead.

## Usage

`data/config/K1Camera.yaml` contains the following information under `cameras`:

- `topic`: the DDS topic carrying `sensor_msgs/Image`. This is the ROS 2 topic path prefixed with
  `rt/`, so ROS 2 topic `/boostercamera/head/rgb` becomes `rt/boostercamera/head/rgb`
- `name`: name of the camera, used to differentiate between outputs
- `id`: camera ID to differentiate between images
- `lens`: optional fallback lens parameters, used only until `sensor_msgs/CameraInfo` arrives on
  `<topic>/camera_info`. `focal_length` and `centre` are normalised by the image width. A warning is
  logged if no `CameraInfo` has arrived five seconds after startup

Topics are subscribed to once at startup. Changing `topic` in the config while running will not
re-point the subscriptions; restart the binary instead.

## Consumes

- `message::input::Sensors` to buffer recent `Hcw` transforms

## Emits

- `message::input::Image`

## Dependencies

- [Booster Robotics SDK](https://github.com/BoosterRobotics/booster_robotics_sdk)
- [OpenCV](https://opencv.org/)
- A running BoosterOS publishing the configured camera topics
