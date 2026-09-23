# PlanSavePlayback

## Description

Replays a goalie recording into `planning::PlanSave`, so its decisions on a recorded shot can be stepped through in a debugger (roles/data/plansaveplayback.role).

It drives `nbs::Player` in `SEQUENTIAL` mode by default. The player holds NUClear's clock still and emits the next message only once everything the last one triggered has finished. So a breakpoint pauses the replay too, and when it continues nothing has moved on. Live against NUSim, the simulator would keep running while NUbots sat at the breakpoint. It also emits the `Save` task that the recording's harness (`tools::GoalieShotBenchmark`) asked for live.

The replay is open loop: PlanSave recomputes its decisions from the recorded inputs, but its `Block` commands have no provider and go nowhere, so nothing it does changes what comes next.

## Usage

1. Record with `roles/nusim/goalieshots.role` and `support::logging::DataLogging`, logging at least the messages in `PlanSavePlayback.yaml`. Log `message.planning.SavePlan` too, to compare the replayed decisions with the live ones.
2. Enable the role (`./b configure --set_roles data-plansaveplayback`), build it, and run it with the recording:

   ```sh
   ./b run data/plansaveplayback recordings/goalieshots/<file>.nbs
   ```

   Recording paths are relative to the build directory, which has a `recordings` link. Without any, it plays `files` from `PlanSavePlayback.yaml`.
3. To debug it, attach VS Code to the NUbots container and run the generated **Plansaveplayback: data** launch configuration. Set the recording in `files` in `PlanSavePlayback.yaml` rather than in the launch configuration's `args`, because `./b configure` regenerates `.vscode/launch.json`.

`message.localisation.Ball` is replayed straight into PlanSave, whatever it was when recorded: the filter's estimate with `ground_truth_ball: estimate`, or NUSim's true ball with `true_state` or `true_crossing`. To step through the ball filter as well, add `localisation::BallLocalisation` to the role, replay `message.vision.Balls` instead of `message.localisation.Ball`, and record an `estimate` run.

## Consumes

- `NUClear::message::CommandLineArguments`: the recordings to play
- `message::nbs::player::PlaybackFinished`

## Emits

- `message::planning::Save` Task
- `message::nbs::player::SetModeRequest`, `LoadRequest`, `PlayRequest`

## Dependencies

- `nbs::Player`, `extension::Director` and `planning::PlanSave`
