# GoalieShotBenchmark

## Description

Rolls shots at the goalie in NUSim and scores them against NUSim's ground truth. `planning::PlanSave` guards the goal through the full stack:

NUSim camera → YOLO → ball UKF → PlanSave → `skill::K1BlockPolicy` (CUSTOM mode) → NUSim.

It measures what the mjlab envelope can't see: vision and filter delay, the hand-off from the walk to the block policy, and PlanSave's decisions.

The harness does what `purpose::Goalie` does while it defends. It emits `Save` above a `WalkToFieldPosition` back to the home spot, keeps the head on the ball, and runs fall recovery. It emits `Save` only while a shot is set up or rolling. Between shots it withdraws it, so the walk takes the goalie home: PlanSave positions the goalie against any ball it sees, and on NUSim's true ball (`true_state`, `true_crossing`) that includes the ball parked out of play.

For each shot it:

1. Rests the ball in front of the goal for `settle_time`, so the ball estimate converges and PlanSave picks GUARD (ball within `guard_distance`) or IDLE. In IDLE PlanSave walks the goalie from home towards its chosen spot, and it may still be on its way at the kick; `robot_dx`, `robot_dy` and `robot_yaw` in `shots.csv` say where it got to.
2. Kicks it through a point on the goalie's home line over NUSim's ball command. The distribution is the mjlab goalkeeper task's "full" level: crossing ±0.8 m, speed 1.5–4 m/s, from 2–4.5 m.
3. Follows the ball until it crosses the goal line (a goal between the posts, a miss outside them), stops, or `max_roll_time` passes.
4. Parks the ball out of play, and waits for the goalie to be upright, home and facing the field before the next shot.

A save is an on-target shot that doesn't go in, as in mjlab.

## Usage

```sh
# NUSim (branch tumminello/goalie-shooter-test or later): the goalie keyframe stands the robot at home
./b run sim/soccer --headless --keyframe goalie
# NUbots_K1
./b run nusim/goalieshots
```

Results are written to `recordings/goalie_shot_benchmark/<timestamp>/shots.csv`, one row per shot, with:
- the shot and the true `dy` and time to arrival from the goalie at the kick (the envelope's axes);
- where the goalie stood;
- PlanSave's mode at the kick, and its first BLOCK tick: how long after the kick, the predicted `dy` and σ, `P(on target)`, and the envelope's expected save rate;
- how close the ball came, whether the goalie fell, and the outcome.

Each shot is also logged, with a summary at the end.

With `ground_truth_field: true` (the default), the harness publishes `Field` from NUSim's ground truth, and the role has no field localisation. On NUSim's symmetric field, localisation can settle on the wrong end, and the walk then takes the goalie away from its goal. That tests localisation, not saving.

`ground_truth_ball` picks which ball the stack runs on, so ground truth can be swapped for the estimates one piece at a time to find where the goalie loses its shots. The harness emits it as `message::booster::NUSimBallSource`:

| `ground_truth_ball` | Ball state (everything that reads `localisation::Ball`) | Where the shot crosses (PlanSave) |
|---|---|---|
| `estimate` (default) | vision → `BallLocalisation`'s filter | PlanSave's rolling-ball prediction |
| `true_state` | NUSim's true ball, published by `BallLocalisation` with zero covariance | PlanSave's prediction, from the true state |
| `true_crossing` | as `true_state` | NUSim's forecast: the ball rolled ahead in a copy of the scene without the robot (NUSim `rt/nusim/gt/ball_crossing/*`) |

`true_crossing` needs NUSim from `tumminello/goalie-shooter-test` at `68e1e4f` or later. Each run's choice is in the `ball_source` column of `shots.csv`.

In a 4-shot `true_crossing` run (23 Sep 2026), all 4 on-target shots were saved, and PlanSave blocked a median 0.03 s after the kick. `true_dy` in `shots.csv` is still the rolling model's crossing at the kick, from where the goalie stood then. So it differs from PlanSave's `plan_dy` for a shot kicked while the goalie was still walking home (0.2–0.5 m in the two such shots).

## Consumes

- `message::booster::NUSimBallGroundTruth`, `NUSimRobotGroundTruth` (`input::NUSimGroundTruth`)
- `message::planning::SavePlan`
- `message::input::Sensors`, `message::support::FieldDescription`

## Emits

- `message::booster::NUSimBallCommand`: place, kick and park the ball
- `message::booster::NUSimBallSource`: which ball the stack runs on (`ground_truth_ball`)
- `message::localisation::Field` (ground-truth mode), or `PenaltyReset` to seed field localisation at home
- Tasks: `Save`, `WalkToFieldPosition`, `LookAtBall`, `LookAround`, `FallRecovery`

## Dependencies

- NUSim with the ball command and ground-truth topics, and the `goalie` keyframe
- `planning::PlanSave`, `skill::K1BlockPolicy` and the rest of `roles/nusim/goalieshots.role`

## Known issues

- **Vision sees false balls on the goalie's own body.** After the real ball is parked out of play, the ball filter can re-acquire one of them beside the goalie. PlanSave now ignores a ball that isn't in front of the goalie, so this no longer holds the goalie in place, but the false detections are still there.
- **NUSim physics blow-ups.** The harness notices the robot teleport when NUSim resets itself, marks the shot `sim_reset`, leaves it out of the summary, and logs the last 2 s of ground truth and ball commands. Every blow-up seen so far came from ball commands whose velocity and spin were left unset. `NUSimBallCommand`'s Eigen vectors aren't zeroed by default, so NUSim got uninitialised memory. The harness now sets them explicitly, and NUSim refuses non-finite or absurd commands.
- **The binary can abort while shutting down** after the last shot (`shutdown_when_done`): "double free or corruption", then SIGABRT. It happens after `shots.csv` is closed and the summary logged, so results are unaffected. It hasn't been tracked down yet.
