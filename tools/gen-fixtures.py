#!/usr/bin/env python3
"""Regenerate the tiny ONNX test fixtures committed under test/data/.

Usage: python3 tools/gen-fixtures.py test/data
Requires the Python `onnx` package (build-time only, never a package
dependency of the Racket package): pip install onnx
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = sys.argv[1] if len(sys.argv) > 1 else "test/data"
os.makedirs(OUT, exist_ok=True)

OPSET = 13  # old enough for any runtime we target, new enough for Add broadcasting


def save(model, name):
    path = os.path.join(OUT, name)
    # ORT <= 1.19 (the oldest runtime our CI exercises) supports IR <= 10;
    # pin so the same fixtures load everywhere.
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    print("wrote", path, os.path.getsize(path), "bytes")


# --- 1. add.onnx: Z = X + K, X float32[2,2], K = [[1,2],[3,4]] ---
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 2])
K_init = numpy_helper.from_array(
    np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32"), "K")
add_graph = helper.make_graph(
    [helper.make_node("Add", ["X", "K"], ["Z"])],
    "add_graph",
    [X],
    [Z],
    initializer=[K_init],
)
save(helper.make_model(add_graph, opset_imports=[helper.make_opsetid("", OPSET)]),
     "add.onnx")

# --- 2. gemm.onnx: Y = A x W + B; A float32[1,3], W float32[3,2], B float32[2] ---
A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 3])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
W_init = numpy_helper.from_array(
    np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32"), "W")
B_init = numpy_helper.from_array(
    np.array([10.0, 20.0], dtype="float32"), "B")
gemm_graph = helper.make_graph(
    [helper.make_node("Gemm", ["A", "W", "B"], ["Y"],
                      alpha=1.0, beta=1.0, transA=0, transB=0)],
    "gemm_graph",
    [A],
    [Y],
    initializer=[W_init, B_init],
)
save(helper.make_model(gemm_graph, opset_imports=[helper.make_opsetid("", OPSET)]),
     "gemm.onnx")

# --- 3. add-dynamic.onnx: same Add graph but first dim of X is symbolic ("N") ---
# onnx.helper accepts an int (dim_value) or a str (dim_param) per shape entry.
# The initializer is [1,2] (broadcast over dim 0) on purpose: a [2,2]
# initializer would let ORT's shape inference unify N with 2 and replace
# the symbolic dimension with a concrete one at load time.
K_dyn = numpy_helper.from_array(np.array([[1.0, 2.0]], dtype="float32"), "K")
Xd = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 2])
Zd = helper.make_tensor_value_info("Z", TensorProto.FLOAT, ["N", 2])
dyn_graph = helper.make_graph(
    [helper.make_node("Add", ["X", "K"], ["Z"])],
    "add_dynamic_graph",
    [Xd],
    [Zd],
    initializer=[K_dyn],
)
save(helper.make_model(dyn_graph, opset_imports=[helper.make_opsetid("", OPSET)]),
     "add-dynamic.onnx")

print("expected results:")
print("  add.onnx          X=[[1,1],[1,1]]         -> Z=[[2,3],[4,5]]")
print("  gemm.onnx         A=[[1,2,3]]             -> Y=[[32,48]]")
print("  add-dynamic.onnx  X=[[0.5,0.5]]           -> Z=[[1.5,2.5]]  (broadcasts over N)")
