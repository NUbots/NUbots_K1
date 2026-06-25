# FieldLocalisation

## Description

This module provides a robust, probabilistic Particle Filter (Monte Carlo Localisation) to estimate the robot's pose on the soccer field. It tracks the transformation from the odometry-drifting `world` frame to the absolute `field` frame (`Hfw`).

## Configuration

- `num_particles`: The total number of particles to use. (Default: 1000)
- `random_particle_injection_rate`: Proportion of particles replaced with random uniform samples at each resampling step to handle kidnapped-robot problems.
- `process_noise`: Noise coefficients applied during odometry updates to account for drift.
- `*_sigma`: Measurement likelihood sensitivities for field lines, intersections, and goal posts.

## Dependencies

- `Sensors`: Provides the `Hrw` odometry updates.
- `FieldLines`, `FieldIntersections`, `Goals`: Used to compute particle likelihoods against the map.
- `FieldDescription`: Generates the internal `OccupancyMap`.

## Emits

- `message::localisation::Field`: Contains the `Hfw` transform, covariance, uncertainty score, and particle cloud.
