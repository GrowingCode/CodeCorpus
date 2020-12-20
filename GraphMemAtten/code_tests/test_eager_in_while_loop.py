import tensorflow as tf
from meta_info.non_hyper_constant import int_type


def t_cond(i, i_len, *_):
  return tf.less(i, i_len)

def t_body(i, i_len, t_info):
  ti = t_info[i]
  if ti.numpy() == 0:
    print("0")
  else:
    print("1")
  return i+1, i_len, t_info

i = tf.constant(0, int_type)
t_info = tf.convert_to_tensor([0,1,1,1,0,0,0])
i_len = tf.constant(len(t_info), int_type)
_, _, _ = tf.while_loop(t_cond, t_body, [i, i_len, t_info])









