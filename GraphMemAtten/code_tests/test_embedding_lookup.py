import tensorflow as tf
from meta_info.non_hyper_constant import int_type


s_1 = tf.ones([1,5], int_type)
s_2 = s_1 + 1
s_3 = s_2 + 1
s_4 = s_3 + 1
s_5 = s_4 + 1

s = tf.concat([s_1,s_2,s_3,s_4,s_5],axis=0)
i = tf.constant([[1,2],[3,4]])
ss = tf.nn.embedding_lookup(s,i)
print(ss)

