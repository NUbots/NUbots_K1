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
- `message::booster::BoosterHeadRot` for the head targets (the policy does not own the head)
- `message::booster::BoosterModeState` (cached, optional) — logged with every observation so
  a capture can be segmented by the mode the robot was actually in

## Emits

- `message::actuation::K1Servos` (22 motors, SDK `JointIndexK1` serial order) as a subtask
- `message::booster::BoosterMode` (CUSTOM) when the walk task starts
- `message::behaviour::state::WalkState`

## Contract

Trained in the NUbots [mjlab fork](https://github.com/NUbots/mjlab), branch `k1`, task
`Mjlab-Velocity-Flat-Booster-K1`. The shipped checkpoint is W&B run `2p0glew7`, iteration
14999 (`data/k1_walk_mjlab_2p0glew7_14999.onnx`).

### ONNX graph I/O

| Tensor | Name | Shape | Type |
|---|---|---|---|
| Input | `obs` | `[1, 1775]` | `float32` |
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

### Observation layout (71 floats per frame)

| Offset | Count | Field | Notes |
|---|---|---|---|
| 0  | 3  | Angular velocity (gyro), body frame | rad/s, from `RawSensors::gyroscope`. |
| 3  | 3  | Projected gravity, body frame | world `(0,0,-1)` rotated into the trunk frame by the firmware attitude estimate; unit vector (upright ⇒ `(0,0,-1)`). |
| 6  | 20 | `q − default_pose` | policy joints only, rad. |
| 26 | 20 | `dq` | policy joints only, rad/s, **unscaled**. |
| 46 | 20 | `last_action` | previous **raw** network output, before scaling and before the joint-range clamp. |
| 66 | 3  | Command `[vx, vy, wz]` | body-frame planar velocity (m/s, m/s, rad/s), passed through as-is. |
| 69 | 2  | Gait clock `[sin 2πφ, cos 2πφ]` | see below. |

Total `3 + 3 + 20 + 20 + 20 + 3 + 2 = 71`.

No per-term scaling and no clipping: every mjlab observation term has scale 1.0.

**No base linear velocity.** There is no measured base linear velocity on the real K1 in
CUSTOM mode, so it is a critic-only privileged quantity in training and nothing here
estimates it.

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

### Gait clock

A fixed-period clock the policy does not control, indexed on episode time in training:
`φ = (t / gait_period) mod 1`, `gait_period = 0.6 s` (exactly 30 control steps at 50 Hz).
The same clock drove the swing-height and contact-mismatch rewards, so the observed period
must match the trained one.

**Standing gate:** when the command magnitude `‖[vx, vy]‖ + |wz| ≤ command_threshold`
(0.05), the observed clock collapses to `(0, 0)` — off the unit circle, *not* pinned to a
phase. That is a distinct "standing" input rather than a walking phase, and it is what
turns the gait rewards off in training. The internal phase keeps advancing regardless.

Here the phase advances on the **measured** loop period, not a hardcoded 0.02 s, so the gait
keeps its trained wall-clock rate when the loop runs slow.

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

The training curriculum's final envelope is `vx ∈ [-0.6, 1.2]`, `vy ∈ [-0.4, 0.4]`,
`wz ∈ [-1.0, 1.0]`. `PlanWalkPath`'s ball-adjust mode can ask for `wz = 1.5`, which is
outside it. Commands are passed through unclipped, as in training.

## Instrumentation

At `log_level: TRACE` every tick emits `WALKOBS <tick> mode=<K1Mode> t=<s> dt=<s>` followed
by all 71 observations of the newest frame. `tools/analysis/segment_walk_log.py` splits a
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
