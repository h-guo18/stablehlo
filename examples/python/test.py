  tests = [
    # No numpy types for f8 - skipping fp8 tests
    ("f16", np.asarray(1, np.float16)),
    ("f32", np.asarray(2, np.float32)),
    ("f64", np.asarray(3, np.double)),
    ("1xi8", np.asarray([4], np.int8)),
    ("1xi16", np.asarray([5], np.int16)),
    ("1xi32", np.asarray([-6], np.int32)),
    # Numpy's uint treated as int by DenseElementsAttr, skipping np.uint tests
    ("2x2xf16", np.asarray([1, 2, 3, 4], np.float16).reshape(2,2)),
    ("2x1x2xf16", np.asarray([1, 2, 3, 4], np.float16).reshape(2,1,2)),
  ]
  for test in tests:
    tensor_type, arg = test
    with ir.Context() as context:
      stablehlo.register_dialect(context)
      m = ir.Module.parse(ASM_FORMAT.format(tensor_type))
      args = [ir.DenseIntElementsAttr.get(arg)]

    actual = np.array(stablehlo.eval_module(m, args)[0])
    expected = arg + arg
    assert (actual == expected).all()
