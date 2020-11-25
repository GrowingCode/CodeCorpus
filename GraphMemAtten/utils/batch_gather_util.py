import tensorflow as tf
from meta_info.non_hyper_constant import int_type


def batch_gather(contents, ids):
  ''' content shape: [all_len, batch_size, feature_size] '''
  ''' ids shape: [part_len, batch_size] '''
  batch_size = tf.shape(ids)[1]
  col = tf.range(batch_size)
  one_col = tf.expand_dims(col, axis=0)
#   print(one_col)
  part_len = tf.shape(ids)[0]
  all_col = tf.tile(one_col, [part_len, 1])
#   print(all_col)
  r_all_col = tf.expand_dims(all_col, axis=2)
  r_ids = tf.expand_dims(ids, axis=2)
  nd_ids = tf.concat([r_ids, r_all_col], axis=2)
#   print(nd_ids)
  outputs = tf.gather_nd(contents, nd_ids)
  ''' outputs shape: [part_len, batch_size, feature_size] '''
  return outputs
  

if __name__ == '__main__':
  sp1 = tf.ones([1, 1, 10], int_type)
  sp2 = tf.ones([1, 1, 10], int_type) + 100
  s1 = tf.concat([sp1, sp2], axis=1)
  s2 = s1 + 1
  s3 = s2 + 1
  s4 = s3 + 1
  s5 = s4 + 1
  
  contents = tf.concat([s1, s2, s3, s4, s5], axis=0)
  print(contents)
  ids = tf.constant([[1,2],[2,4]], int_type)
  outputs = batch_gather(contents, ids)
  print(outputs)



