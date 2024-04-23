# Copyright 2024 The StableHLO Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mlir import ir
import mlir.dialects.stablehlo as stablehlo
import mlir.dialects.func as func
from mlir.ir import Context, Location, InsertionPoint, Module
import numpy as np


myir = """
func.func @test(%arg0: tensor<{0}>) -> tensor<{0}> {{
  %0 = stablehlo.add %arg0, %arg0 : (tensor<{0}>, tensor<{0}>) -> tensor<{0}>
  func.return %0 : tensor<{0}>
}}
""".format("f32")
with Context() as ctx:
  stablehlo.register_dialect(ctx)
  # module = Module.create()
  module = Module.parse(myir)

  # with InsertionPoint(module.body):
  #   @func.func()
  #   def main():
  #     a_value = ir.DenseElementsAttr.get(np.zeros(shape=[3,4], dtype=np.int64))
  #     b_value = ir.DenseElementsAttr.get(np.zeros(shape=[3,4], dtype=np.int64))
  #     a = stablehlo.constant(a_value)
  #     b = stablehlo.constant(b_value)
  #     add = stablehlo.add(a, b)
  #     # stablehlo.ReturnOp(add.owner)
  #     return add
      # stablehlo.ReturnOp(add.owner)
    

  args = [ir.DenseIntElementsAttr.get(np.asarray(2, np.float32))]
res = np.array(stablehlo.eval_module(module,args))
print(res)
print(str(module))
