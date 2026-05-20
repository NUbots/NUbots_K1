# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NUbots is a humanoid soccer robot project for the RoboCup competition, running on the Booster K1 platform. The system is built around **NUClear Roles**, a custom CMake-based reactive actor framework where C++ modules communicate exclusively via typed message passing.

## Build Commands

All builds run inside Docker (`nubots/nubots:generic`). The `./b` wrapper invokes `nuclear/b.py` via `uv run`.

```bash
# Configure (run once, or when CMakeLists change)
./b configure --build-dir build

# Build everything
./b build --build-dir build

# Build with parallelism
./b build --build-dir build -j8

# Run all C++ unit tests
cd build && ninja test

# Check code formatting (C++, CMake, Python, JS/TS)
./b format --check --verbose

# Apply formatting
./b format
```

## NUsight2 (Frontend) Commands

```bash
cd nusight2
yarn install         # Install dependencies
yarn start           # Dev server
yarn test            # Run Vitest unit tests
yarn tscheck         # TypeScript type checking
yarn eslint --fix    # ESLint
yarn format          # Prettier
```

## Architecture

### NUClear Reactor Pattern

Every functional unit is a **reactor** — a C++ class inheriting `NUClear::Reactor`. Reactors never call each other directly; they `emit<>()` typed messages and bind `on<>()` triggers. The framework dispatches messages to all reactors that subscribed to that type.

A role file (e.g. `roles/robocup.role`) lists the set of modules to compile into one binary and their initialization order.

### Message Pipeline

```
Input → Vision → Localisation → Strategy → Planning → Skill → Actuation → Platform
```

- **input/**: Camera frames, GameController UDP packets, IMU/sensor data
- **vision/**: Ball detector (VisualMesh, YOLO), field line detection
- **localisation/**: Kalman-filter-based position estimation
- **strategy/**: High-level decisions (WalkToBall, FindObject)
- **planning/**: Motion trajectories, head scan patterns
- **skill/**: Walk engine, KickWalk, GetUp, Look
- **actuation/**: Inverse kinematics, servo command generation
- **platform/**: Booster K1 hardware abstraction (subcontroller communication)

All data types exchanged between modules are defined as Protocol Buffers in `shared/message/`.

### Module Structure

```
module/<category>/<Name>/
├── src/<Name>.hpp        # Reactor class declaration
├── src/<Name>.cpp        # Implementation
├── CMakeLists.txt        # Registers module with NUClear Roles
├── data/                 # YAML config files (deployed alongside binary)
└── test/                 # Google Test unit tests
```

To add a new module: create the directory structure, define the reactor class, and add it to a `.role` file.

### Shared Code

- `shared/message/` — all protobuf `.proto` definitions; CMake auto-generates C++ headers
- `shared/utility/` — math helpers, coordinate transforms, Kalman filter, vision utilities
- `shared/extension/` — behaviour tree DSL, configuration management (`Configuration<>` DSL)

### NUsight2 (Visualization)

React/TypeScript frontend that connects to a running robot or recording via NUClear networking. Lives entirely in `nusight2/`. The backend (`src/server/`) speaks NBS/NUClear protocol; the client (`src/client/`) renders robot state in-browser.

## Key Conventions

- C++20, 120-character line limit, Google-style clang-format
- Warnings are errors in CI (`-Wall -Wpedantic -Wextra -Werror`)
- Config values are loaded via `Configuration<ModuleName>` — YAML files live in `module/.../data/`
- Proto messages use `NUClear::message::` namespace after codegen
- Role files control which modules are active; a module absent from a role is never instantiated
- `nuclear/` is a git subtree — do not edit it directly unless updating the subtree
