import tensorflow as tf
from meta_info.non_hyper_constant import gradient_clip_abs_range


def clip_gradients(grads):
  final_grads = []
  for (gv, var) in grads:
    if gv is not None:
      grad = tf.clip_by_value(gv, -gradient_clip_abs_range, gradient_clip_abs_range)
      final_grads.append((grad, var))
  return final_grads







