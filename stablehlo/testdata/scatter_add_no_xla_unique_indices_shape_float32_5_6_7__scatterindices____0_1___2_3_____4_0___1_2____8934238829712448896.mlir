// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xf32>, tensor<5x2x2xf32>)
    %2 = call @expected() : () -> tensor<5x6x7xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xf32>, tensor<2x2x2xi32>, tensor<5x2x2xf32>) -> tensor<5x6x7xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xf32>, tensor<5x6x7xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xf32>, tensor<5x2x2xf32>) {
    %0 = stablehlo.constant dense<"0x104216BF4E5647C0D80CC2C061EE37BE87186EC0A9928540FF33944008D2F83EA36BE5BF7B1A8AC0E785C1BF610406409656883F1C80664055E474BFB55BB0BC4670EC3E6500363FF5667840D9FA4740E3A5363FC82F9C3FC9F6E93EC0B475C01AB5A03C84B018C06CD8F33F4F5002C0FBBDE1C0004B84BEF7DB0F40378B29C0E1405EC0BD9B823F9AD73B40EA6EC8BF3C1D84BD36A1FE3D6DEAFFBCB7783040635ADE3EDC1D5C409EED2E4010EE813F13EEC1C0260C0540EC9764C02326763E4EB03E3FD1C0DEBF98595EC087DCC940204C9BC009CA36402415743F407B1EBF9BED29C03840263F8CA780BAAAF783C0F7AF37C07EE61AC0C1930B3F755780408E3874C07DC9A7C078DD8D40089F83409C98CCBF05C8A73F6137A940D44EF5BDE6D043C030676F40A3967140C1C4C13EB49810C0C307E53ECBAB223F40DC08C0E81F7140F3B249BE5F5A99BE800DB040DFB482BF64286E3FAE73EDBF80CA98BF4CE18A402D64B4BF74675FBDC29B033F12788240D4C80CC05ADD7B3F113691C069BFE9BE21AA2D3FB8B4863F40C79FBE5C4DB7BF90B6BC3F125C74BF83C30F409756294094A493BF7B2F0CC0E1CD4F40B87D53405D0E70C0EB7AF03FB55F553FCE3A0E40E5BC13C0FFF72D3FDAEB9140E2D783C02205EABF5DD19DBEBBC29040A86563BF423276BF6295733E0F99164024CCBAC0D499603FC0EEBCBF9DCB0340923926407FB96840F040B640BF17A7BF4592D3BFFF58FD402D359ABF3F7F943EB3DC5DBF5D9CBD4016642C40D6A2C8405B16A1C0248A963F3D4A8C4082C71DC0A1B1E83F9E796AC03B8B85BF456B443FDB9490404E983D3F509E3CC0D5DF3640BDDF55C096A25DC0824196C0AA5D7140D005BD3FCA91B2BF2D951F40D7FC6E4098A4B3C02330303E78D16CC0BA55D8BF3FEE443FA255613E9FC680C0526928C0ED29763E0778BC402C372940F05B5740CFD5A53FEEABD7BF4F9BA3BDBC24423F2A710A3F2159D93F488773BF09C60A40EA036F40DA897840346C92BC3CB4B5BDA713DC3E02E13CBF1BA0363FBC763D4058FEEE3E1399D13F75C5FE3E759A3B3F3D2E5DC07B65414059F9DA3F89960DC03F690E3DC18A9D3F6F0D823F124645BF963800BF75DEA43E68F609C04000BF3EEF2E33BF199E03C03D97BEC0554A14C066C8B03FC789B13E"> : tensor<5x6x7xf32>
    %1 = stablehlo.constant dense<[[[-0.599414051, -0.594364285], [3.45534658, -1.93522108]], [[1.99793494, -0.206108138], [-9.65725994, -2.63190603]], [[-1.29876268, -2.62840319], [2.19652629, -2.8342886]], [[2.42427158, 0.331273824], [-1.04580593, 3.79799318]], [[-3.83247495, 4.14213943], [0.775025904, -3.658290e+00]]]> : tensor<5x2x2xf32>
    return %0, %1 : tensor<5x6x7xf32>, tensor<5x2x2xf32>
  }
  func.func private @expected() -> tensor<5x6x7xf32> {
    %0 = stablehlo.constant dense<"0x104216BF1BB36DC0D80CC2C061EE37BE87186EC0A9928540FF33944008D2F83EA36BE5BFD007C8C0E785C1BF610406409656883F1C80664055E474BFB55BB0BC4670EC3E18C1EE3DF5667840D9FA4740E3A5363FC82F9C3FC9F6E93EC0B475C01AB5A03C84B018C06CD8F33F4F5002C0905766C0004B84BEF7DB0F40378B29C0E1405EC0BD9B823F9AD73B40EA6EC8BF3C1D84BD36A1FE3D6DEAFFBCB7783040635ADE3EDC1D5C409EED2E4032D5404013EEC1C0260C0540EC9764C02326763E4EB03E3FD1C0DEBF98595EC0E8476B40204C9BC009CA36402415743F407B1EBF9BED29C03840263F8CA780BA1A908AC0F7AF37C07EE61AC0C1930B3F755780408E3874C07DC9A7C078DD8D40089F83409C98CCBF05C8A73FE5D08BC0D44EF5BDE6D043C030676F40A3967140C1C4C13EB49810C0C307E53ECBAB223F40DC08C0E81F7140F3B249BE5F5A99BE800DB040DFB482BFA4A6BCBEAE73EDBF80CA98BF4CE18A402D64B4BF74675FBDC29B033F12788240E816A1C05ADD7B3F113691C069BFE9BE21AA2D3FB8B4863F40C79FBE5C4DB7BFF4B893BF125C74BF83C30F409756294094A493BF7B2F0CC0E1CD4F40B87D53405D0E70C0EB7AF03FB55F553F58678D40E5BC13C0FFF72D3FDAEB9140E2D783C02205EABF5DD19DBEBBC29040A86563BF423276BF6295733E0F99164024CCBAC0D499603FC0EEBCBF70798F40923926407FB96840F040B640BF17A7BF4592D3BFFF58FD402D359ABF1DD18240B3DC5DBF5D9CBD4016642C40D6A2C8405B16A1C0248A963F3D4A8C40EB9308C0A1B1E83F9E796AC03B8B85BF456B443FDB9490404E983D3F509E3CC0D5DF3640BDDF55C096A25DC0C0B8B7C0AA5D7140D005BD3FCA91B2BF2D951F40D7FC6E4098A4B3C02330303E78D16CC0BA55D8BF3FEE443FA255613E9FC680C0526928C0ED29763EC9A803402C372940F05B5740CFD5A53FEEABD7BF4F9BA3BDBC24423F2A710A3FB7E9FABF488773BF09C60A40EA036F40DA897840346C92BC3CB4B5BDA713DC3E90E059401BA0363FBC763D4058FEEE3E1399D13F75C5FE3E759A3B3F3D2E5DC07B65414059F9DA3F89960DC0AD4E4F3FC18A9D3F6F0D823F124645BF963800BF75DEA43E68F609C04000BF3EEF2E33BF199E03C03D97BEC0554A14C066C8B03FC789B13E"> : tensor<5x6x7xf32>
    return %0 : tensor<5x6x7xf32>
  }
}
