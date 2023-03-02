// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xui8>
    %1 = call @expected() : () -> tensor<3xui8>
    %2 = stablehlo.constant dense<255> : tensor<ui8>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xui8>, tensor<ui8>) -> tensor<3xui8>
     reducer(%arg0: tensor<ui8>, %arg1: tensor<ui8>)  {
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xui8> {
    %0 = stablehlo.constant dense<[[5, 2, 4], [6, 10, 3]]> : tensor<2x3xui8>
    return %0 : tensor<2x3xui8>
  }
  func.func private @expected() -> tensor<3xui8> {
    %0 = stablehlo.constant dense<[5, 2, 3]> : tensor<3xui8>
    return %0 : tensor<3xui8>
  }
}