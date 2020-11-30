import tensorflow as tf
from meta_info.hyper_parameter import n_layer


def get_recent_fixed_length_memory(mems, mem_len):
  new_mems = []
  for j in range(n_layer):
    slice_start = tf.maximum(0, tf.shape(mems[j])[0]-mem_len)
    new_mems.append(tf.slice(mems[j], [slice_start, 0, 0], [-1, -1, -1]))
  return new_mems


def update_recent_fixed_length_memory(mems, cat_mems):
  new_mems = []
  for j in range(n_layer):
    new_mems.append(tf.concat([mems[j], cat_mems[j]], axis=0))
  return new_mems








