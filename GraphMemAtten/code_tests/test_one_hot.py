import tensorflow as tf
from meta_info.non_hyper_constant import int_type


res = tf.one_hot(tf.constant([[0,1],[1,2]], int_type), 3)
print(res)




