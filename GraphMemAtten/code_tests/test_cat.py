import tensorflow as tf
from meta_info.non_hyper_constant import int_type


t_cat = tf.zeros([10, 0], int_type) - 1
print(t_cat)

