import tensorflow as tf
from meta_info.non_hyper_constant import int_type, float_type


valid_mask = tf.constant([[0,1],[1,0]], int_type)
nll = tf.constant([[2.0,4.0],[6.0,8.0]], float_type)
r_nll = tf.where(tf.equal(valid_mask, 1), nll, tf.zeros_like(nll))
z_nil = nll * tf.cast(valid_mask, float_type)

print(r_nll)
print(z_nil)



