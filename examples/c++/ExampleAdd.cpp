/* Copyright 2022 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Api.h"

int main() {
  std::vector<float> values{12,23,42,12};
  std::vector<float> values2{24,46,84,24};
  mlir::MLIRContext context;

  /** create module **/
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  module->getContext()->loadDialect<mlir::func::FuncDialect>();
  module->getContext()->loadDialect<mlir::stablehlo::StablehloDialect>();
  module->getContext()->loadDialect<mlir::quant::QuantizationDialect>();
  module->setName("test_module");

  /** create function **/
  // create function argument and result types.
  auto tensorType =
      mlir::RankedTensorType::get({2, 2}, mlir::FloatType::getF32(&context));
  auto func_type =
      mlir::FunctionType::get(&context, {tensorType, tensorType}, {tensorType});

  // create the function and map arguments.
  llvm::ArrayRef<mlir::NamedAttribute> attrs;
  auto function = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&context),
                                             "main", func_type, attrs);
  function.setVisibility(mlir::func::FuncOp::Visibility::Public);
  module->push_back(function);

  // create function block with add operations.
  mlir::Block* block = function.addEntryBlock();
  llvm::SmallVector<mlir::Value, 4> arguments(block->args_begin(),
                                              block->args_end());
  mlir::OpBuilder block_builder = mlir::OpBuilder::atBlockEnd(block);
  mlir::Location loc = block_builder.getUnknownLoc();

//   llvm::SmallVector<mlir::NamedAttribute, 10> attributes;
  mlir::Operation* op =
      block_builder.create<mlir::stablehlo::AddOp>(loc, arguments[0],arguments[0])
          .getOperation();
  mlir::Operation* op2 =
      block_builder.create<mlir::stablehlo::AddOp>(loc, arguments[1],op->getResult(0))
          .getOperation();
  block_builder.create<mlir::func::ReturnOp>(loc, op2->getResult(0));

  /** verify and dump the module **/
  assert(mlir::succeeded(mlir::verify(module.get())));

  /* interpret the function "main" with concrete inputs **/
//     auto getConstTensor = [&](float val) {
//     // std::vector<float> values(2 * 2, val); // Fill the tensor with `val`
//     return mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef<float>(values));
//   };

  auto inputValue1 = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef<float>(values));
  auto inputValue2 = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef<float>(values));
  auto expectedValue = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef<float>(values2));

  mlir::stablehlo::InterpreterConfiguration config;
  (*module).dump();
  auto results = evalModule(*module, {inputValue1, inputValue2}, config);
  if ((*results)[0] == expectedValue) printf("YESYESYES\n");
  return failed(results) || (*results)[0] != expectedValue;
}
