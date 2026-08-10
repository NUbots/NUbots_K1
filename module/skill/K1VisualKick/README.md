# K1VisualKick

## Description

Sends a command to the Booster Robotics SDK to initiate a visual kick, along with a kick
reference to aim it. Starting the visual kick switches the robot into a mode that accepts kick
references on the SDK's `rt/kick_ball` topic, so the reference is sent immediately after the
start command.

## Usage

Emit a Kick message when wanting to kick.

## Consumes

- `message::skill::Kick` a request to kick. The kick's `direction` sets the kick direction and power; `leg` is unused as the SDK chooses its own leg
- `message::input::Sensors` used with `Field` to compute the robot's field-space pose for the kick reference
- `message::localisation::Field` used with `Sensors` to compute the robot's field-space pose for the kick reference
- `message::support::FieldDescription` used to compute the goal target position in field space (offset shared with WalkToBall's config)

## Emits

- `message::booster::BoosterVisualKick` which the Booster `HardwareIO` forwards to the SDK via `booster_client.VisualKick(start, version)`.
- `message::booster::BoosterKick` which the Booster `HardwareIO` publishes to the SDK's `rt/kick_ball` topic, carrying the robot's field-space pose and the goal target to aim at.

## Dependencies

- Director
