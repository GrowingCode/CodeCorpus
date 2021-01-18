import tensorflow as tf
from utils.cartesian_util import cartesian_add_one_dim_vector,\
  cartesian_concat_two_dim_mats
from meta_info.non_hyper_constant import top_ks, int_type, float_type


def dp_compute_en_seqs_from_distinct_parallel_tokens(o_log_probs, o_ens):
  ''' o_log_probs and o_ens are both of shape: [steps, top_ks[-1]] '''
  def compute_ens_cond(i, i_len, *_):
    return tf.less(i, i_len)
  
  def compute_ens_body(i, i_len, acc_log_probs, acc_ens):
    o_prob = o_log_probs[i]
    o_en = o_ens[i]
    
    fa_log_probs = cartesian_add_one_dim_vector(acc_log_probs, o_prob)
    fa_ens = cartesian_concat_two_dim_mats(acc_ens, tf.expand_dims(o_en, axis=1))
    
    _, indices = tf.nn.top_k(fa_log_probs, top_ks[-1])
    sorted_probs = tf.gather(fa_log_probs, indices)
    sorted_ens = tf.gather(fa_ens, indices)
    
    return i+1, i_len, sorted_probs, sorted_ens
  
  seq_len = tf.shape(o_log_probs)[0]
  acc_log_probs = o_log_probs[0] # tf.zeros([top_ks[-1]], float_type)
  acc_ens = tf.expand_dims(o_ens[0], axis=1) # tf.zeros([top_ks[-1], 0], int_type)
  _, _, acc_log_probs, acc_ens = tf.while_loop(compute_ens_cond, compute_ens_body, [tf.constant(1, int_type), seq_len, acc_log_probs, acc_ens], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([top_ks[-1]]), tf.TensorShape([top_ks[-1], None])], parallel_iterations=1)
  return acc_log_probs, acc_ens


if __name__ == '__main__':
  ''' test1 '''
  o_log_probs = tf.constant([[0,1,2,3,4,5,6,7,8,9],[29,18,7,6,5,16,3,2,1,0]], float_type)
  o_ens = tf.constant([[0,1,2,3,4,5,6,7,8,9],[9,8,7,6,5,4,3,2,1,0]], int_type)
  res1 = dp_compute_en_seqs_from_distinct_parallel_tokens(o_log_probs, o_ens)
  print("str(res1):" + str(res1))





