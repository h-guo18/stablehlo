from mlir import ir
import mlir.dialects.stablehlo as stablehlo
import mlir.dialects.func as func
from mlir.ir import Context, Location, InsertionPoint, Module
from mlir.execution_engine import ExecutionEngine
import numpy as np


operation_set = set()

with Context() as ctx, Location.unknown():
  stablehlo.register_dialect(ctx)
  module = Module.create()
  with open("alexnet.mlir")as f:
      mlir_str = f.read()
      module = module.parse(mlir_str)
      
  for op in module.body.operations[0].body.blocks[0].operations:
    print(op.name,len(op.operands))
    if op.name not in operation_set:
      operation_set.add(op.name)
    for i,o in enumerate(op.operands):

      print(f"operands {i}:",o.type.element_type,o.type.shape)
      print("\n\n")
      



