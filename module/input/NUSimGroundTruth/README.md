# NUSimGroundTruth

## Description

Bridges NUSim's test surface into NUClear. See NUSim `mujoco/module/SdkBridge/PROTOCOL.md` §6 for the wire contract.

- **Ground truth in:** `rt/nusim/gt/ball` and `rt/nusim/gt/robot` (`nav_msgs/Odometry`, world-frame pose and twist, stamped with the wall clock of the physics snapshot) become `message::booster::NUSimBallGroundTruth` and `NUSimRobotGroundTruth`.
- **Ball crossings in:** `rt/nusim/gt/ball_crossing/robot` and `.../goal` (where the ball will cross the robot's frontal plane and the goal line, from NUSim rolling it ahead without the robot) become `message::booster::NUSimBallCrossings`, both in the robot frame `{r}`.
- **Ball commands out:** `message::booster::NUSimBallCommand` goes to `rt/nusim/ball_command`, which places the ball and sets its velocity, in the simulator world or the robot's yaw frame. It can roll the ball without slipping.

The simulator world `{s}` is the scene's field frame, not the NUbots odometry world `{w}`. Compare estimates through the robot frame `{r}` instead. The bridge does that for the ball: `NUSimBallGroundTruth` also carries `rBRr` and `vBr`, from the latest torso ground truth, and the goal-line crossing is moved into `{r}` the same way.

Only useful against NUSim. On a real robot the topics never appear and the module stays idle.

## Usage

Add `input::NUSimGroundTruth` to a NUSim role, alongside a consumer such as `tools::BallLocalisationBenchmark`.

## Consumes

- `message::booster::NUSimBallCommand`

## Emits

- `message::booster::NUSimBallGroundTruth`
- `message::booster::NUSimRobotGroundTruth`
- `message::booster::NUSimBallCrossings`

## Dependencies

- Booster robotics SDK (DDS channels)
