# K1WalkPolicy

Runs the mujoco_playground K1 joystick walk policy (79-obs / 22-action ONNX) on the
robot side at 50 Hz and streams the resulting joint position targets to
`platform::Booster::HardwareIO` as `message::booster::BoosterLowCmd`, which forwards
them to the Booster SDK `rt/joint_ctrl` topic (honoured in CUSTOM mode).

This replaces `skill::K1Walk`'s `Move()` RPC path: locomotion inference lives in the
NUbots stack, and the robot/simulator only tracks servo joint commands.

## Consumes

- `message::skill::Walk` (Director task) with the target velocity
- `message::platform::RawSensors` for joint feedback, gyro and the IMU attitude
- `message::booster::BoosterHeadRot` for the head targets (the policy does not own the head)
- `message::booster::BoosterModeState` (cached, optional) — logged with every observation
  so a capture can be segmented by the mode the robot was actually in

## Emits

- `message::booster::BoosterLowCmd` (22 motors, SDK JointIndexK1 serial order)
- `message::booster::BoosterMode` (CUSTOM) when the walk task starts
- `message::behaviour::state::WalkState`

## Contract

The observation/action layout is pinned in NUSim `docs/OBS_ACTION_CONTRACT.md`.

**There is no base linear velocity observation.** The 82-obs contract had one, filled by
differentiating `rt/odometer_state` and low-passing the result; the real K1 has no measured
linear velocity in CUSTOM mode, so it is now a critic-only privileged quantity and this
module no longer depends on `BoosterOdometry` at all. The 82-obs checkpoints
(`k1_walk.onnx`, `k1_walk_v1_20260723_617M.onnx`) will not load against this module.

Two things are instrumented for hardware bring-up:

- At `log_level: TRACE` every tick emits `WALKOBS <tick> mode=<K1Mode> t=<s> dt=<s>` plus
  all 79 observations. `tools/analysis/segment_walk_log.py` splits a capture into
  MOVING/FROZEN runs before reporting statistics — the first hardware log was 81.5%
  robot-standing-still, and whole-file statistics from it were misleading enough to be
  quoted as findings.
- Commanded joint positions are clamped to `joint_lower`/`joint_upper` (the trained model's
  ranges) and any clamp is logged at WARN. A saturated action already commands past those
  on several joints; MuJoCo absorbs that silently, a mechanical stop does not.

The gait phase advances on the **measured** loop period, not a hardcoded 0.02 s.
