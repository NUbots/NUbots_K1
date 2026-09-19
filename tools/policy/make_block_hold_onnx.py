#!/usr/bin/env python3
"""Write a hold-stance placeholder ONNX for skill::K1BlockPolicy.

The graph honours the block policy contract v0 (see module/skill/K1BlockPolicy/README.md): input
``obs`` [1, history_window * frame_dim], output ``actions`` [1, n_joints], opset 18. Every action
is zero, so the module commands its default pose. That smoke-tests the hand-off blend, CUSTOM mode,
head tracking and release in NUSim or on the robot before a trained checkpoint exists.

The actions are a zero-weight MatMul of the input rather than a constant, so the graph still
consumes ``obs`` and TensorRT/OpenVINO see the same I/O shapes a real policy has.

Usage:
    python3 tools/policy/make_block_hold_onnx.py [--history-window 1] [--joints 20] [-o PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

COMMAND_DIM = 4  # [active, dy, time_to_arrival, ball_speed]


def frame_dim(n_joints: int) -> int:
    # gyro(3) + projected gravity(3) + q - default, dq, last action (n_joints each) + command
    return 6 + 3 * n_joints + COMMAND_DIM


def build(history_window: int, n_joints: int) -> onnx.ModelProto:
    n_in = history_window * frame_dim(n_joints)
    weights = numpy_helper.from_array(np.zeros((n_in, n_joints), dtype=np.float32), name="zero_weights")
    graph = helper.make_graph(
        nodes=[helper.make_node("MatMul", ["obs", "zero_weights"], ["actions"])],
        name="k1_block_hold_stance",
        inputs=[helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, n_in])],
        outputs=[helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, n_joints])],
        initializer=[weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="NUbots")
    # Match the IR version of opset 18 exports so older runtimes accept it
    model.ir_version = 8
    for key, value in {
        "contract": "k1_block_policy_v0",
        "history_window": str(history_window),
        "frame_dim": str(frame_dim(n_joints)),
        "placeholder": "hold_stance",
    }.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--history-window", type=int, default=1, help="frames in the observation window")
    parser.add_argument("--joints", type=int, default=20, help="number of policy-controlled joints")
    parser.add_argument("-o", "--output", type=Path, help="output path (default: module data dir)")
    args = parser.parse_args()

    output = args.output or (
        Path(__file__).resolve().parents[2]
        / "module/skill/K1BlockPolicy/data"
        / f"k1_block_hold_stance_T{args.history_window}.onnx"
    )
    onnx.save(build(args.history_window, args.joints), output)
    print(f"wrote {output} (obs [1, {args.history_window * frame_dim(args.joints)}] -> actions [1, {args.joints}])")


if __name__ == "__main__":
    main()
