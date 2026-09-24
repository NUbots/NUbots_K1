# K1WalkPolicy

Runs the mjlab K1 velocity-tracking walk policy at 50 Hz on the robot side and streams the
resulting joint position targets as a Director-arbitrated `message::actuation::K1Servos`
subtask, which `platform::Booster::HardwareIO` forwards to the Booster SDK `rt/joint_ctrl`
topic (honoured only in CUSTOM mode).

This replaces `skill::K1Walk`'s `Move()` RPC path: locomotion inference lives in the NUbots
stack, and the robot/simulator only tracks servo joint commands.

## Consumes

- `message::skill::Walk` (Director task) with the target velocity
- `message::platform::RawSensors` for joint feedback, gyro and the IMU attitude
- `message::behaviour::state::Stability` — ticks are skipped while FALLEN, when
  `skill::K1GetUpPolicy` owns the low-level channel
- `message::booster::BoosterOdometryTwist` for the base linear velocity (the controller's `rt/odom`
  twist, body frame)
- `message::booster::BoosterHeadRot` for the head targets (the policy does not own the head)
- `message::booster::BoosterModeState` (cached, optional) — logged with every observation so
  a capture can be segmented by the mode the robot was actually in

## Emits

- `message::actuation::K1Servos` (22 motors, SDK `JointIndexK1` serial order) as a subtask
- `message::booster::BoosterMode` (CUSTOM) when the walk task starts
- `message::behaviour::state::WalkState`

## Contract

Trained in the NUbots [mjlab fork](https://github.com/NUbots/mjlab), branch `k1`, task
`Mjlab-Velocity-Flat-Booster-K1`. The shipped checkpoint is W&B run `t98yksya`
(`k1-noclock-linvel`, trained at mjlab `55b975c96`), iteration 14999
(`data/k1-noclock-linvel.onnx`). It dropped the previous run's (`2p0glew7`) gait clock and
observes the base linear velocity instead.

### ONNX graph I/O

| Tensor | Name | Shape | Type |
|---|---|---|---|
| Input | `obs` | `[1, 1800]` | `float32` |
| Output | `actions` | `[1, 20]` | `float32` |

The module checks these element counts against the configured contract on load and refuses
any graph that does not match, on TensorRT and on the OpenVINO fallback alike. The older
mujoco_playground checkpoints (`k1_walk*.onnx`, 79-obs / 22-action, single frame) are
therefore **not** loadable here.

Both observation normalizers (`EmpiricalNormalization` over the current frame and over the
history frames) are **baked into the exported graph**. The deployment side applies no
normalization of its own, and no action clipping — the training runner had `clip_actions`
unset.

### Observation window

This is an observation-history policy (`mjlab.rl.obs_history`). The graph input is a flat
window of the last **25** observation frames, **time-major, oldest frame first**; a TCN
inside the graph encodes it and the graph slices the current observation out of the last
frame. One frame is byte-for-byte the actor observation vector, so this module keeps a
single ring buffer of the frame it already builds.

On a fresh start the window is seeded by repeating the first frame 25 times, matching the
training-side `CircularBuffer` backfill on reset. The window is also dropped whenever the
next tick cannot continue the previous one — a new walk task, or a resumption after the
get-up policy owned the channel — rather than letting the encoder read a 0.5 s history that
straddles a fall.

**The loop rate is load-bearing twice over.** The window spans 25 control *steps*, so a loop
running at 40 Hz feeds the encoder 0.625 s of history instead of 0.5 s. A loop period off
the trained 0.02 s by more than 5 ms is counted and reported at WARN every 2 s. That is a
bug to fix, not to compensate for.

### Observation layout (72 floats per frame)

| Offset | Count | Field | Notes |
|---|---|---|---|
| 0  | 3  | Base linear velocity, body frame | m/s, from the controller's odometry twist (see below). |
| 3  | 3  | Angular velocity (gyro), body frame | rad/s, from `RawSensors::gyroscope`. |
| 6  | 3  | Projected gravity, body frame | world `(0,0,-1)` rotated into the trunk frame by the firmware attitude estimate; unit vector (upright ⇒ `(0,0,-1)`). |
| 9  | 20 | `q − default_pose` | policy joints only, rad. |
| 29 | 20 | `dq` | policy joints only, rad/s, **unscaled**. |
| 49 | 20 | `last_action` | previous **raw** network output, before scaling and before the joint-range clamp. |
| 69 | 3  | Command `[vx, vy, wz]` | body-frame planar velocity (m/s, m/s, rad/s), passed through as-is. |

Total `3 + 3 + 3 + 20 + 20 + 20 + 3 = 72`.

No per-term scaling and no clipping: every mjlab observation term has scale 1.0.

### Base linear velocity

Training observes the `imu_lin_vel` velocimeter, which sits at the trunk origin with the
trunk's orientation: the trunk's linear velocity in the trunk frame, with 0.05/0.05/0.08 m/s
noise and 0–60 ms of delay. Here it is the Booster controller's odometry twist (`rt/odom`,
`nav_msgs/Odometry`), which `platform::Booster::HardwareIO` emits as
`BoosterOdometryTwist`: its linear part is in the twist's `child_frame_id`, the body frame by
ROS convention. HardwareIO logs both frame ids with the first message, since Booster does not
document them.

A twist older than `linear_velocity_max_age` (0.1 s), or none at all, is observed as zero,
and the module warns every 2 s while that lasts. Zero reads as standing still, so the policy
walks badly on it: a robot or simulator that does not publish `rt/odom` in CUSTOM mode needs
fixing, not ignoring.

### Joint order

The mjlab model's 20 policy joints are the MuJoCo model order with the two head joints
dropped, which is exactly `JointIndexK1` minus its first two slots. Policy index `j` maps to
`JointIndexK1` index `j + 2`, so **no permutation table is needed** — `policy_joints` in the
config is simply `2..21`.

```
 0 HeadYaw            1 HeadPitch          <- not policy controlled
 2 LeftShoulderPitch  3 LeftShoulderRoll  4 LeftElbowPitch  5 LeftElbowYaw
 6 RightShoulderPitch 7 RightShoulderRoll 8 RightElbowPitch 9 RightElbowYaw
10 LeftHipPitch  11 LeftHipRoll  12 LeftHipYaw  13 LeftKneePitch  14 LeftAnklePitch  15 LeftAnkleRoll
16 RightHipPitch 17 RightHipRoll 18 RightHipYaw 19 RightKneePitch 20 RightAnklePitch 21 RightAnkleRoll
```

### Action layout (20 floats) & application

Joint target offsets around the training keyframe:

```
q_ref[j]    = default_pose[j] + action_scale * action_scale_joint[j] * action[j]
q_ref[head] = clamp(latest BoosterHeadRot)
q_ref[else] = default_pose[j]
q_cmd       = clamp(lerp(measured, q_ref, handoff_blend), joint_lower, joint_upper)
```

- `default_pose` is the mjlab `STAND_BENT_KNEES_KEYFRAME`. It differs from the previous
  mujoco_playground pose in exactly one place: the elbow yaws sit at ∓0.15 rather than 0,
  because the elbow-yaw hard stop at 0 lies outside the 0.9 soft joint limit and an
  arms-down pose at 0 would pay the joint-limit penalty every training step.
- `action_scale_joint[j] = 0.25 · effort_limit[j] / kp[j]` (booster_train convention);
  `action_scale` is a global multiplier (1.0 for this policy). The values are unchanged from
  the mujoco_playground config — both training setups use Booster's actuator catalogue.
- The PD gains sent in the LowCmd are the training-time gains: `kp = armature · (2πf)²`,
  `kd = 2ζ · armature · 2πf`, per joint group.
- `handoff_blend` (0.3 s) cross-fades from the measured pose into the policy target on entry
  to CUSTOM. Training always starts at the default pose; deployment starts from wherever
  PREP left the robot.
- `last_action` is the **raw** network output, kept unclamped and unscaled, because that is
  what the observation expects. Only `q_cmd` is clamped.

### Saturation

At `a = ±1` these joints command past their mechanical stops, so the clamp bites and the
count is logged at DEBUG:

| Joint | `JointIndexK1` | Overshoot |
|---|---|---|
| Left/Right Shoulder Roll | 3, 7 | 0.447 rad |
| Left/Right Elbow Yaw | 5, 9 | 0.737 rad |
| Left/Right Hip Roll | 11, 17 | 0.486 rad |
| Left/Right Knee Pitch | 13, 19 | 0.064 rad |

### Command envelope

The training curriculum's final envelope is `vx ∈ [-1.0, 2.0]`, `vy ∈ [-0.8, 0.8]`,
`wz ∈ [-2.0, 2.0]`, reached by iteration 8000. Commands are passed through unclipped, as in
training.

## Instrumentation

At `log_level: TRACE` every tick emits `WALKOBS <tick> mode=<K1Mode> t=<s> dt=<s>` followed
by all 72 observations of the newest frame. `tools/analysis/segment_walk_log.py` splits a
capture into MOVING/FROZEN runs before reporting statistics — the first hardware log was
81.5% robot-standing-still, and whole-file statistics from it were misleading enough to be
quoted as findings.

## Re-exporting the policy

From the mjlab `k1` branch, the runner exports an ONNX on every checkpoint save
(`VelocityOnPolicyRunner.save`), with the contract attached as ONNX metadata props
(`history_window`, `history_obs_dim`, `history_layout`, `joint_names`, `action_scale`,
`default_joint_pos`, `joint_stiffness`, `joint_damping`). To export from an existing
checkpoint, build the play env, load it with `load_cfg={"actor": True}` and call
`export_policy_to_onnx` followed by `attach_metadata_to_onnx`.
