# Surface friction measurements

The walk policy's ground-contact domain randomisation and every NUSim surface sweep are
currently bracketing **guesses**. This file is where the measured numbers go, and until it
has them, no claim about "in distribution" or "out of distribution" for a surface is
supported.

## Why it matters

The one controlled hardware result about the walk policy is that its failure is
surface-dependent: keyboardwalk at 0.2 m/s forward, same policy, same command, flat-footed
and fine on carpet, progressively losing balance on the synthetic-grass field with tip-toe
appearing during the degradation. Everything identical across those two runs is ruled out
as a cause. The variable is foot–ground interaction.

Training randomises the floor's sliding friction log-uniformly over **[0.25, 1.2]**
(`mujoco_playground/_src/locomotion/k1/randomize.py`). If a surface sits outside that, the
policy is being asked to extrapolate. If both surfaces sit comfortably inside it, friction
range is not the story and the compliance axis or the torque envelope is.

## Procedure

Tilt test on a **loaded** foot — the robot's own weight on the foot, not a bare sole, since
the fibres of synthetic grass compress differently under load. Raise the surface until the
foot slides, and record the angle.

```
mu = tan(slip angle)
```

Reference points: `mu = 0.25` slides at **14.0°**, `0.3` at **16.7°**, `0.6` at **31.0°**,
`1.0` at **45.0°**, `1.2` at **50.2°**.

Take at least three trials per surface and record the spread, not just the mean — the
interesting quantity is the low end, because that is where the friction budget runs out.

## Measurements

| Surface | Date | Trial angles (deg) | mu = tan(theta) | Notes |
|---|---|---|---|---|
| Carpet (lab) | | | | walk holds up here |
| Synthetic grass (field) | | | | walk progressively falls here |
| | | | | |

## Contact compliance

Sliding friction is only one axis. Every training episode to date ran on perfectly rigid
ground (`solref` was never randomised), and a NUSim sweep found that soft ground
destabilises the deployed policy independently of friction — at `solref_timeconst = 0.10`
it fell even at `mu = 1.0`, with the strongest toe-down sole tilt of any run
(see `NUSim/docs/SURFACE_FIDELITY.md`).

There is no tilt-test equivalent for this. A usable proxy: press a foot into the surface
with a known load and measure the penetration depth, or record the settling time of a
dropped ball. Training now randomises the floor `solref` over `timeconst` `[0.006, 0.03]`
and `dampratio` `[0.8, 1.5]`; record whatever is measured here so that range can be checked
rather than assumed.

| Surface | Date | Method | Result | In the trained range? |
|---|---|---|---|---|
| Carpet (lab) | | | | |
| Synthetic grass (field) | | | | |
