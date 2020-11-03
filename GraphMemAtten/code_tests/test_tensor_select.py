import tensorflow as tf
from meta_info.non_hyper_constant import float_type


a = tf.constant([[1,2],[3,4]], float_type)

b = a[:,-1]

print(b)


def a():
  return 1, 2, 3, 4

_, second, *_ = a()

print(second)



