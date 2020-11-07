import tensorflow as tf
from meta_info.non_hyper_constant import float_type, uniform_range,\
  normal_stddev

variable_initializer_seed = 13

def random_uniform_variable_initializer(shape, mt=0, bs=0, ini_range=uniform_range):
  if ini_range == 0.0:
    return zero_variable_initializer(shape)
  seed = mt*variable_initializer_seed+bs
  if mt == 0 and bs == 0:
    seed = None
  return tf.random.uniform(shape, minval=-ini_range, maxval=ini_range, dtype=float_type, seed=seed)

def random_normal_variable_initializer(shape, mt=0, bs=0, ini_stddev=normal_stddev):
  if normal_stddev == 0.0:
    return zero_variable_initializer(shape)
  seed = mt*variable_initializer_seed+bs
  if mt == 0 and bs == 0:
    seed = None
  return tf.random.normal(shape, stddev=ini_stddev, dtype=float_type, seed=seed)

def one_variable_initializer(shape):
  return tf.ones(shape, dtype=float_type)

def zero_variable_initializer(shape):
  return tf.zeros(shape, dtype=float_type)

