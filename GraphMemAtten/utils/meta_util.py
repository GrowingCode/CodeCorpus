import tensorflow as tf
from meta_info.hyper_parameter import n_layer, d_model, top_ks


def get_varied_memory_shape_in_while_loop():
  mems_shapes = []
  for _ in range(n_layer):
    mems_shapes.append(tf.TensorShape([None, top_ks[-1], d_model]))
  return mems_shapes




