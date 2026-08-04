# NUSimCamera

## Description

Reads the camera image out of the shared-memory segment written by **NUSim** (the
MuJoCo simulator's `Camera` module / `SharedImageWriter`) and emits it as
`message::input::Image`. This is the simulation-only counterpart of `input::K1Camera`:
K1Camera reads the on-robot [NUbridge](https://github.com/NUbots/NUbridge) writer,
which uses the VERSION 1 `interprocess_mutex` + condition layout; NUSimCamera reads
NUSim's VERSION 2 **lock-free seqlock** layout.

The seqlock has no shared lock, so a behaviour binary killed while reading cannot wedge
the segment for the next one — you can restart the behaviour without restarting NUSim.
That is why the sim uses a separate reader instead of changing K1Camera: the on-robot
NUbridge writer stays on the v1 layout, so the real-robot path is untouched.

Use this module in `roles/nusim/*`; use `input::K1Camera` in the on-robot roles.

## Usage

`data/config/NUSimCamera.yaml` lists, under `cameras`:
- `segment`: name of the [Boost](https://www.boost.org/) shared-memory segment NUSim writes
- `id`: camera ID to differentiate between images
- `name`: name of the camera, used to differentiate between outputs

## Emits

- `message::input::Image`

## Dependencies

- [Boost](https://www.boost.org/)
- NUSim writing the camera shared-memory segment (`SharedImageWriter`, VERSION 2 seqlock)
