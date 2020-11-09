import tensorflow as tf
from meta_info.non_hyper_constant import int_type


s1 = tf.ones([1, 5, 10], int_type)
s2 = s1 + 1
s3 = s2 + 1
s4 = s3 + 1
s5 = s4 + 1

s = tf.concat([s1, s2, s3, s4, s5], axis=0)
ids = tf.constant([[1,2],[3,4]], int_type)

ss1 = tf.gather_nd(s, ids)
print(ss1)

ss2 = tf.nn.embedding_lookup(s, ids)
print(ss2)










