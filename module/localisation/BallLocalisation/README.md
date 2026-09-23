# BallLocalisation

## Description

This module takes in a list of vision balls, uses the ball measurement closest to our current estimate and applies an
Unscented Kalman Filter to estimate the balls position and velocity in world space.

## Usage

Include this module to allow the robot to estimate the balls position and velocity.

## Consumes

- `message::vision::Balls` uses the ball position estimate from vision
- `message::input::Sensors` uses sensors to compute transform from camera {c} to torso space {t}
- `message::support::FieldDescription` uses field description to obtain height of ball off the ground
- `message::booster::NUSimBallSource`, `NUSimBallGroundTruth` (NUSim only): while a harness sets the source to ground truth (`tools::GoalieShotBenchmark`'s `ground_truth_ball`), NUSim's true ball is published instead of the filter's estimate, with zero covariance. The filter keeps running but its estimate isn't emitted. On a real robot neither message exists and nothing changes.

## Emits

- `message::localisation::Ball` contains filtered ball position measurement
