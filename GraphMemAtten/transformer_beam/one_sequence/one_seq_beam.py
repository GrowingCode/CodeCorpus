import tensorflow as tf
from utils.cartesian_util import batch_cartesian_add_each_scalar_in_vect,\
  batch_cartesian_concat_one_dim_vect_and_each_scalar_in_vect
from meta_info.non_hyper_constant import int_type, float_type
from meta_info.hyper_parameter import oracle_mem_len, top_ks, n_layer,\
  oracle_tgt_len
from transformer.model import transformer
from utils.meta_util import get_varied_memory_shape_in_while_loop


class OneSeqBeam:
  
  def __init__(self):
    pass
  
  def __call__(self, mems_before_last, last_token, whole_seq, skt_token_split):
    
    ''' generate stored memory for whole sequence '''
    seq_len = tf.shape(whole_seq)[0]
    
    def mem_gen_cond(idx, *_):
      return tf.less(idx, seq_len)
    
    def mem_gen_body(idx, *all_mems):
      seq_part_size = tf.minimum(idx + oracle_tgt_len, seq_len) - idx
      dec_inp = tf.slice(whole_seq, [idx, 0], [seq_part_size, -1])
      target = tf.zeros_like(dec_inp) - 1
      
      new_all_mems = all_mems
      
      probs, predictions, _, new_mems = transformer(dec_inp, target, mems, is_training=0, mem_len=oracle_mem_len)
      
      return (idx + oracle_tgt_len, *new_all_mems)
    
    def osb_cond(i, i_len, *_):
      return tf.less(i, i_len)
  
    def osb_body(i, i_len, ):
      o_skt_start = skt_token_split[i][0]
      o_skt_end = skt_token_split[i][1]
      o_token_end = skt_token_split[i][2]
      pass
    
    tf.while_loop(osb_cond, osb_body, [0, tf.shape(skt_token_split)[0], ], shape_invariants, parallel_iterations=1)
  
  def infer(self, last_token, mems_before_last, steps):
    
    ''' here mems_before_last shape must be [n_layer memory_length 1 feature_size] '''
    ''' here mems_before_last shape should be extended to [n_layer memory_length batch_size feature_size] '''
    for i in range(n_layer):
      mems_before_last[i] = tf.tile(mems_before_last[i], [1, top_ks[-1], 1])
    
    def infer_cond(i, i_len, *_):
      return tf.less(i, i_len)
    
    def infer_body(i, i_len, probs, ens, l_token, *mems_tuple):
      ''' in the following, batch_size should be top_ks[-1]]] '''
      ''' probs shape should be [batch_size] '''
      ''' ens shape should be [batch_size, varied_with_max_length_(steps + 1)] '''
      ''' l_token shape should be [1, batch_size] '''
      mems = list(mems_tuple)
      ''' mems shape should be [n_layer memory_length batch_size feature_size] '''
      probs, predictions, _, new_mems = transformer(l_token, tf.zeros_like(l_token)-1, mems, is_training=0, mem_len=oracle_mem_len)
      ''' probs          should be [1, batch_size, top_ks[-1]] '''
      ''' predictions    should be [1, batch_size, top_ks[-1]] '''
      r_probs = tf.squeeze(probs, [0])
      r_predictions = tf.squeeze(predictions, [0])
      ''' r_predictions should be [batch_size, top_ks[-1]] '''
      
      new_probs = batch_cartesian_add_each_scalar_in_vect(probs, r_probs)
      new_ens = batch_cartesian_concat_one_dim_vect_and_each_scalar_in_vect(ens, r_predictions)
      ''' new_probs should be [batch_size * top_ks[-1]] '''
      ''' new_ens should be [batch_size * top_ks[-1], varied_with_max_length_steps + 1] '''
      
      probs_values, probs_indices = tf.nn.top_k(new_probs, top_ks[-1])
      ens_values = tf.gather(new_ens, probs_indices)
      
      ''' update memory '''
      mems_indices = tf.math.floordiv(probs_indices, top_ks[-1])
      for i in range(n_layer):
        new_mems[i] = tf.gather(new_mems[i], mems_indices, axis=1)
      
      last_col_ens = tf.expand_dims(ens_values[:, -1], axis=0)
      return (i+1, i_len, probs_values, ens_values, last_col_ens, *new_mems)
    
    i = tf.constant(0, int_type)
    i_len = tf.constant(steps, int_type)
    probs = tf.zeros([top_ks[-1]], float_type)
    ens = tf.zeros([top_ks[-1], 1], int_type) - 1
    mems_shapes = get_varied_memory_shape_in_while_loop()
    _, _, probs, ens, *_ = tf.while_loop(infer_cond, infer_body, loop_vars=[i, i_len, probs, ens, last_token, *mems_before_last], shape_invariants=[tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([top_ks[-1]]), tf.TensorShape([top_ks[-1], None]), tf.TensorShape([1, top_ks[-1]]), *mems_shapes], parallel_iterations=1)
    ens = tf.slice(ens, [0, 1], [-1, -1])
    return ens
  

