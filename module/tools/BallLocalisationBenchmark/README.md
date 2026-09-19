# BallLocalisationBenchmark

## Description

Validates `localisation::BallLocalisation`, the ball UKF, against NUSim ground truth.

The robot stands still and looks at the ball. Through `input::NUSimGroundTruth`, NUSim rolls the ball at it from a seeded random schedule. Each shot:
1. Places the ball in front of the robot.
2. Lets it rest for `settle_time`, so the estimate converges.
3. Kicks it: same position, new velocity, rolling without slipping, towards a point on the robot's frontal plane.

Every UKF estimate is compared with the ground truth at the moment it becomes available to consumers. Every raw vision detection is compared with the ground truth at its image time, which separates detector error from filter error. All comparisons are in the robot frame `{r}`, which is immune to odometry drift.

For each shot, over the roll:
- **Position and velocity error:** RMSE and maximum.
- **Velocity response:** the time from the kick until the estimated velocity is within 10% of the true speed and within 20° of the true direction. This measures how quickly the filter follows a kick.
- **Effective lag:** the delay at which the estimate best matches the delayed ground truth, and the error at that lag.
- **Detector error:** raw detection error against the ground truth.
- **Vision latency:** the median time from image capture to the `Balls` message.
- **Rates:** estimate and detection rates.

Results go to `output_dir/<timestamp>/`:
- `samples.csv`: every estimate.
- `detections.csv`: every detection.
- `summary.csv`: one row per shot.

The end of the run logs a summary line. Analyse the results with `tools/analysis/ball_localisation_report.py`.

## Usage

```sh
./b run nusim/ballvalidation   # against a running NUSim
```

Shot ranges, count, seed and timing are in `BallLocalisationBenchmark.yaml`. With `shutdown_when_done` the binary exits after the last shot.

**Things that bias the results:**
- **Hardware for vision:** without a GPU, YOLO runs on OpenVINO CPU and NUSim renders in software. Detection rate and vision latency are then far worse than on the robot's Orin with TensorRT, so only trust rates and latencies from a GPU run.
- **Ball size:** the detector turns apparent size into range using `FieldDescription.ball_radius`. The module warns if NUSim's ball differs from it by more than 5 mm.

## Consumes

- `message::booster::NUSimBallGroundTruth`, `NUSimRobotGroundTruth`: ground truth.
- `message::localisation::Ball`: the estimate under test.
- `message::vision::Balls`: raw detections.
- `message::input::Sensors`: `Hrw`, to put estimates in the robot frame.
- `message::support::FieldDescription`: the ball-radius check.

## Emits

- `message::booster::NUSimBallCommand`: the shots.
- `message::strategy::StandStill`, `message::strategy::LookAtBall`, `message::planning::LookAround` Tasks.

## Dependencies

- `input::NUSimGroundTruth` and a running NUSim.
