from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops

@ops.RegisterGradient("AvgPoolGrad")
def _AvgPoolGradGrad(op, grad):
  return (array_ops.stop_gradient(op.inputs[0]), gen_nn_ops._avg_pool(
      grad,
      op.get_attr("ksize"),
      op.get_attr("strides"),
      op.get_attr("padding"),
      data_format=op.get_attr("data_format")))
