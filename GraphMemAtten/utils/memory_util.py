import tensorflow as tf
from meta_info.hyper_parameter import n_layer


def get_recent_fixed_length_memory(mems, mem_len):
  new_mems = []
  curr_mem_len = tf.shape(mems[0])[0]
  for j in range(n_layer):
    slice_start = curr_mem_len - mem_len#tf.maximum(0, )
    assert slice_start >= 0
    new_mems.append(tf.slice(mems[j], [slice_start, 0, 0], [-1, -1, -1]))
  return new_mems


def get_specified_varied_length_memory(mems, last_index_in_extracted_mems, mem_len, train_test_consistent):
  new_mems = []
  if last_index_in_extracted_mems == -1:
    last_index_in_extracted_mems = tf.shape(mems[0])[0] - 1
  extracted_with_before_size = last_index_in_extracted_mems + 1
  if train_test_consistent:
    e_size = mem_len + tf.math.mod(extracted_with_before_size, mem_len)
  else:
    e_size = mem_len
    # tf.maximum(0, )
  slice_start = extracted_with_before_size - e_size
  slice_length = last_index_in_extracted_mems - slice_start + 1
  assert slice_length == e_size
  assert slice_start >= 0
  for j in range(n_layer):
    new_mems.append(tf.slice(mems[j], [slice_start, 0, 0], [slice_length, -1, -1]))
  return new_mems


def update_recent_fixed_length_memory(mems, cat_mems):
  new_mems = []
  for j in range(n_layer):
    new_mems.append(tf.concat([mems[j], cat_mems[j]], axis=0))
  return new_mems









