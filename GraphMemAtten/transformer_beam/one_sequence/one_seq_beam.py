import tensorflow as tf
from utils.cartesian_util import batch_cartesian_add_each_scalar_in_vect,\
  batch_cartesian_concat_one_dim_vect_and_each_scalar_in_vect,\
  cartesian_add_one_dim_vector, cartesian_concat_two_dim_mats
from meta_info.non_hyper_constant import int_type, float_type
from meta_info.hyper_parameter import oracle_mem_len, top_ks, n_layer,\
  oracle_tgt_len, accuracy_based_on_whole, oracle_predict_mem_len, beam_mode,\
  standard_beam, multi_infer
from transformer.model import transformer
from utils.meta_util import get_varied_memory_shape_in_while_loop
from utils.memory_util import get_recent_fixed_length_memory,\
  update_recent_fixed_length_memory
from utils.accuracy_util import compute_accuracy_of_sequences
from utils.loss_accurate_util import compute_loss_and_accurate_and_top_k_prediction_from_linear_with_computed_embeddings


class OneSeqBeam():
  
  def __init__(self, transformer_model, multi_position_transfer):
    self.transformer_model = transformer_model
    self.multi_position_transfer = multi_position_transfer
  
  def __call__(self, mems, last_token_before_whole_seq, whole_seq, skt_token_split):
    ''' note that mems include last_token_before_whole_seq '''
    origin_mems_len = tf.shape(mems[0])[0]
    
    ''' generate stored memory for whole sequence '''
    seq_len = tf.shape(whole_seq)[0]
    
    def mem_gen_cond(idx, *_):
      return tf.less(idx, seq_len)
    
    def mem_gen_body(idx, *all_mems):
      all_mems = list(all_mems)
      seq_part_size = tf.minimum(idx + oracle_tgt_len, seq_len) - idx
      dec_inp = tf.slice(whole_seq, [idx, 0], [seq_part_size, -1])
      target = tf.zeros_like(dec_inp) - 1
      
      temp_mems = get_recent_fixed_length_memory(all_mems, oracle_mem_len)
      
      _, _, _, _, new_mems = transformer(dec_inp, target, temp_mems, is_training=0, mem_len=oracle_mem_len)
      new_all_mems = update_recent_fixed_length_memory(all_mems, new_mems)
      
      return (idx + oracle_tgt_len, *new_all_mems)
    
    _, all_mems = tf.while_loop(mem_gen_cond, mem_gen_body, [tf.constant(0, int_type), *mems], parallel_iterations=1)
    
    def osb_cond(i, i_len, *_):
      return tf.less(i, i_len)
  
    def osb_body(i, i_len, skt_each_acc, skt_whole_acc, skt_count, token_each_acc, token_whole_acc, token_count):
      o_skt_start = skt_token_split[i][0]
      o_skt_end = skt_token_split[i][1]
      o_token_end = skt_token_split[i][2]
      
      skt_last_before = o_skt_start - 1
      skt_last_before_valid = tf.cast(tf.greater_equal(skt_last_before, 0), int_type)
      skt_real_last_before = tf.stack([last_token_before_whole_seq, skt_last_before])[skt_last_before_valid]
      skt_part_seq = tf.slice(whole_seq, [o_skt_start, 0], [o_skt_end-o_skt_start+1, -1])
      
      skt_mems_end = origin_mems_len + skt_last_before
      skt_mems_start = tf.maximum(skt_mems_end - oracle_predict_mem_len, 0)
      skt_mems_before_last = []
      for i in range(n_layer):
        skt_mems_before_last.append(tf.slice(all_mems, [skt_mems_start, 0, 0], [skt_mems_end-skt_mems_start+1, -1, -1]))
      skt_f_each_acc, skt_f_whole_acc, skt_f_count = self.infer_and_compute_accuracy(skt_mems_before_last, skt_real_last_before, skt_part_seq)
      skt_each_acc += skt_f_each_acc
      skt_whole_acc += skt_f_whole_acc
      skt_count += skt_f_count
      
      token_last_before = o_skt_end
      token_real_last_before = token_last_before
      token_part_seq = tf.slice(whole_seq, [o_skt_end+1, 0], [o_token_end-o_skt_end, -1])
      
      token_mems_end = origin_mems_len + token_last_before
      token_mems_start = tf.maximum(token_mems_end - oracle_predict_mem_len, 0)
      token_mems_before_last = []
      for i in range(n_layer):
        token_mems_before_last.append(tf.slice(all_mems, [token_mems_start, 0, 0], [token_mems_end-token_mems_start+1, -1, -1]))
      token_f_each_acc, token_f_whole_acc, token_f_count = self.infer_and_compute_accuracy(token_mems_before_last, token_real_last_before, token_part_seq)
      token_each_acc += token_f_each_acc
      token_whole_acc += token_f_whole_acc
      token_count += token_f_count
      
      return i+1, i_len, skt_each_acc, skt_whole_acc, skt_count, token_each_acc, token_whole_acc, token_count
    
    i = tf.constant(0, int_type)
    i_len = tf.constant(tf.shape(skt_token_split)[0], int_type)
    skt_each_acc, skt_whole_acc, skt_count = tf.constant(0, float_type), tf.constant(0, float_type), tf.constant(0, int_type)
    token_each_acc, token_whole_acc, token_count = tf.constant(0, float_type), tf.constant(0, float_type), tf.constant(0, int_type)
    _, _, skt_each_acc, skt_whole_acc, skt_count, token_each_acc, token_whole_acc, token_count = tf.while_loop(osb_cond, osb_body, [i, i_len, skt_each_acc, skt_whole_acc, skt_count, token_each_acc, token_whole_acc, token_count], parallel_iterations=1)
    return skt_each_acc, skt_whole_acc, skt_count, token_each_acc, token_whole_acc, token_count
  
  def infer_and_compute_accuracy(self, mems_before_last, last_token_before_part_seq, part_seq):
    if beam_mode == standard_beam:
      inferred_ens = self.infer(mems_before_last, last_token_before_part_seq, tf.shape(part_seq)[0])
    elif beam_mode == multi_infer:
      inferred_ens = self.multi_infer(mems_before_last, last_token_before_part_seq, tf.shape(part_seq)[0])
    else:
      assert False
    f_each_acc, f_whole_acc, f_count = compute_accuracy_of_sequences(inferred_ens, part_seq, compute_one_whole=accuracy_based_on_whole)
    return f_each_acc, f_whole_acc, f_count
  
  def infer(self, mems_before_last, last_token, steps):
    
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
      _, probs, predictions, _, new_mems = transformer(l_token, tf.zeros_like(l_token)-1, mems, is_training=0, mem_len=oracle_mem_len)
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
  
  def multi_infer(self, mems_before_last, last_token, steps):
    output, _, _, _, _ = transformer(last_token, tf.zeros_like(last_token)-1, mems_before_last, is_training=0)
    ''' output shape should be [predict_length batch_size feature_size] '''
    ''' output shape should be [1 1 feature_size] '''
    
    def multi_infer_cond(self, i, i_len, *_):
      return tf.less(i, i_len)
    
    def multi_infer_body(self, i, i_len, o_log_probs, o_ens):
      TODO
      t_h = self.multi_position_transfer.transfer(i, output)
      
      o_log_probs_of_this_node, o_ens_of_this_node, _, _, _ = compute_loss_and_accurate_and_top_k_prediction_from_linear_with_computed_embeddings(False, self.transformer_model.get_token_output_parameters(), -1, t_h)
      
      o_log_probs = tf.concat([o_log_probs, [o_log_probs_of_this_node]], axis=0)
      o_ens = tf.concat([o_ens, [o_ens_of_this_node]], axis=0)
      
      return (i+1, i_len, o_log_probs, o_ens)
    
    i = tf.constant(0, int_type)
    o_log_probs = tf.zeros([0, top_ks[-1]], float_type)
    o_ens = tf.zeros([0, top_ks[-1]], int_type)
    _, _, o_log_probs, o_ens = tf.while_loop(multi_infer_cond, multi_infer_body, [i, steps, o_log_probs, o_ens], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, top_ks[-1]]), tf.TensorShape([None, top_ks[-1]])], parallel_iterations=1)
    computed_en_seqs = dp_compute_en_seqs_from_distinct_parallel_tokens(o_log_probs, o_ens)
    return computed_en_seqs
    
    
def dp_compute_en_seqs_from_distinct_parallel_tokens(o_log_probs, o_ens):
  
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
  return acc_ens
  
  












