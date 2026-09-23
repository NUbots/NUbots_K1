# PlanSave

## Description

The goalie's save planner. It predicts shots from the ball estimate and runs the block policy (`skill::K1BlockPolicy`) when one is coming. It uses the policy's measured **capability envelope** to say how likely the block is to work.

Every tick (50 Hz) it:

1. **Predicts:** takes our own ball estimate (`rBWw`, `vBw` and their covariance) and rolls it forward with constant rolling deceleration, to where it crosses:
   - the goalie's frontal plane (x = 0 in the robot frame `{r}`): `dy`, time to arrival, and their standard deviations;
   - our goal line: where along it, and `P(on target)`, the probability the ball goes between the posts.

   The prediction is `ShotCommand._predict_crossing` from the mjlab goalkeeper task, because the policy was trained on commands computed that way. The covariance is carried through a numerical Jacobian. `kick_widening` can widen the velocity covariance for the first moments of a shot. It is off: through the full stack in NUSim, the filter's covariance at the commit was already far wider than the actual error.
2. **Scores the block:** averages the envelope's conservative save rate (a Wilson lower bound per cell) over the Gaussian on `dy`, at the predicted time and speed. Probability outside the measured grid counts as a miss.
3. **Chooses:**
   - **IDLE:** emits nothing, so the positioning walk (a lower-priority sibling of `Save`) runs.
   - **GUARD:** the ball is within `guard_distance` and at least `guard_min_ahead` in front of the goalie's line. Emits an inactive `Block`, so the goalie holds the policy's ready stance in CUSTOM mode, which is the state the envelope was measured from.
   - **BLOCK:** a shot reaches the goalie's line within `max_time` and has `P(on target) ≥ min_p_on_target`. Emits `Block{active, dy, t, v}` every tick. The command follows the training rule: active while the ball is on its way, zeros otherwise. PlanSave sticks with the block until the shot is over: the ball is no longer on its way (stopped, past the goalie or going away) or lost, for `release_delay`.

   The block is the only skill so far. PlanSave still blocks a shot whose expected success is under `block_threshold`, but warns: that is the gap a dive policy would fill.

Each tick is published as `message::planning::SavePlan`: the prediction, its uncertainty, `P(on target)`, the expected save rate and fall rate, and the mode. Shots are also logged at INFO when they start and end, so the predictions can be matched against the outcomes.

## The envelope and its policy

`data/config/SaveEnvelope.yaml` holds the envelope. It counts on-target shots, saves and falls over signed `dy` × time to arrival × ball speed. It records the SHA-256 of the ONNX it was measured from. PlanSave hashes the file named by `model_path` in `K1BlockPolicy.yaml`. **If the two don't match, or either is missing, PlanSave never blocks, and the goalie only positions.** Replacing the policy therefore means re-measuring its envelope:

```sh
# In mjlab, at the commit the policy was trained at
python -m mjlab.tasks.goalkeeper.scripts.measure_envelope --checkpoint logs/rsl_rl/k1_block/<run>/model_<n>.pt --num-envs 1024 --steps 6000
# Here
python3 tools/policy/make_save_envelope.py <run>/envelope.csv -o module/planning/PlanSave/data/config/SaveEnvelope.yaml --name "<run>"
```

The current envelope is goalkeeper run 12 (wandb `zrhfjas8`, `model_2999`, measured at mjlab `0dd3d95f0`): 26,245 shots from the task's "full" level. Crossings are within ±0.8 m of the keeper, speeds 1.5–4 m/s, from 2–4.5 m. It saved 69.9% of the on-target shots inside the grid, and fell during 5.9%. `measure_envelope` writes shots that end in a fall as off target, so `make_save_envelope.py` counts them back in as failed on-target shots (see its docstring).

## Usage

- Add `planning::PlanSave` and `skill::K1BlockPolicy` to the role. PlanSave reads `K1BlockPolicy.yaml`.
- Emit a `message::planning::Save` Task while the goalie guards the goal. Give it a higher priority than the positioning walk, so a `Block` it emits takes the servos from the walk. `purpose::Goalie` does this while it defends. `purpose::Tester` has `plan_save_priority`.

## Consumes

- `message::planning::Save`: Director task, guard the goal.
- `message::localisation::Ball`: our own estimate (confidence > 0) with covariance. Teammates' balls carry no velocity and are ignored.
- `message::input::Sensors` (`Hrw`), `message::localisation::Field` (`Hfw`), `message::support::FieldDescription`.
- `message::booster::NUSimBallSource`, `NUSimBallCrossings` (NUSim only): with the source at `TRUE_CROSSING`, the crossings of the goalie's line and our goal line come from NUSim rolling the ball ahead without the robot, not from the prediction, with zero σ. `SavePlan.true_crossing` says so. Without a forecast from the last `ball.timeout`, the ball counts as invalid rather than falling back to the prediction. On a real robot neither message exists.

## Emits

- `message::skill::Block` Task: an inactive command (GUARD) or the live shot command (BLOCK).
- `message::planning::SavePlan`: the tick's prediction and decision.

## Dependencies

- Director
- `skill::K1BlockPolicy` and its config
- A ball filter that publishes covariance (`localisation::BallLocalisation` from `tumminello/ball-ukf-fixes` on)

## Results in NUSim

`tools::GoalieShotBenchmark` (roles/nusim/goalieshots.role) rolled 40 shots from the mjlab "full" level at the goalie through the full stack, on 22 Sep 2026. With ground-truth field localisation, it saved 28 of 40 on-target shots (70%, against 68% in mjlab for run 12), with no falls. PlanSave blocked all 40, a median 0.23 s after the kick (10–90%: 0.17–0.31 s). At the commit, `dy` was off by 0.05 m RMS, against a σ of 0.93 m. So the envelope's expected saves (17 of 40) are conservative, mostly from the too-wide σ.

## Not done yet

- **The envelope only covers starting from the ready stance.** A shot that arrives while the goalie is walking also pays for the mode switch and hand-off, which nothing has measured yet. `guard_distance` is the stopgap.
- **Positioning doesn't use the envelope yet.** `purpose::Goalie` still follows the ball's y along the goal.
- **σ(dy) is ~20× wider than the actual error at the commit.** That delays the commit (P(on target) needs time to reach `min_p_on_target`) and makes expected success conservative. The width comes from the ball filter's kick handling.
- **Vision sees balls on the goalie's own feet and hands in NUSim**, and the ball filter can lock onto one and miss a real shot. `guard_min_ahead` stops PlanSave guarding them, but the filter lock is a vision and localisation fix.
- **No penalty-kick handling.** Nothing here moves before the ball is touched, but PlanSave doesn't know the game state.
