import tensorflow as tf
from meta_info.non_hyper_constant import float_type, int_type, top_ks,\
  top_ks_tensors


def compute_accuracy_of_sequences(computed_en_seqs, oracle_computed_en_seq, compute_one_whole=True):
  
  e_lens = tf.ones_like(oracle_computed_en_seq)
  eq_all_lens_int = tf.reduce_sum(e_lens)
  e_lens = tf.cast(e_lens, float_type)
  eq_all_lens = tf.reduce_sum(e_lens)
  
  def compute_acc_cond(i, i_len, *_):
    return i < i_len
  
  def compute_acc_body(i, i_len, epos_acc, whole_acc):
    one_computed_en_seq = computed_en_seqs[i]
    tc = tf.zeros([tf.shape(oracle_computed_en_seq)[-1] - tf.shape(one_computed_en_seq)[-1]], int_type) - 1
    r_one_computed_en_seq = tf.concat([one_computed_en_seq, tc], axis=0)
    
    eq = tf.cast(tf.equal(r_one_computed_en_seq, oracle_computed_en_seq), float_type)
    eq_lens = tf.reduce_sum(eq * e_lens)
#     eq_all_acc = tf.reduce_sum(eq)
#     eq_all_count = tf.cast(tf.shape(eq)[-1], float_type)
    if compute_one_whole:
      epos_right = eq_lens / eq_all_lens
      whole_right = tf.cast(tf.equal(eq_lens, eq_all_lens), float_type)
    else:
      epos_right = eq_lens
      whole_right = tf.cast(tf.equal(eq_lens, eq_all_lens), float_type) * eq_all_lens
    
    epos_r_acc = tf.maximum(epos_acc[-1], epos_right)
    whole_r_acc = tf.maximum(whole_acc[-1], whole_right)
    
    epos_acc = tf.concat([epos_acc, [epos_r_acc]], axis=0)
    whole_acc = tf.concat([whole_acc, [whole_r_acc]], axis=0)
    
    return i+1, i_len, epos_acc, whole_acc
  
  n = tf.shape(computed_en_seqs)[0]
  each_acc = tf.zeros([1], float_type)
  whole_acc = tf.zeros([1], float_type)
  _, _, each_acc, whole_acc = tf.while_loop(compute_acc_cond, compute_acc_body, [0, n, each_acc, whole_acc], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None]), tf.TensorShape([None])])
  
  f_each_acc = tf.zeros([0], float_type)
  f_whole_acc = tf.zeros([0], float_type)
  acc_len = tf.shape(each_acc)[-1]
  tpk_len = len(top_ks)
  for i in range(tpk_len):
    tpk = top_ks[i]
    r_sel = tf.minimum(tpk, acc_len-1)
    f_each_acc = tf.concat([f_each_acc, [each_acc[r_sel]]], axis=0)
    f_whole_acc = tf.concat([f_whole_acc, [whole_acc[r_sel]]], axis=0)
  
  if compute_one_whole:
    f_count = tf.constant(1, int_type)
  else:
    f_count = eq_all_lens_int
  
  return f_each_acc, f_whole_acc, f_count


def compute_batch_top_ks_accuracy(predictions, oracle_tgt, valid_mask):
  ''' predictions shape: [seq_length, batch_size, top_ks[-1]] '''
  ''' oracle_tgt shape: [seq_length, batch_size] '''
  ''' valid_mask shape: [seq_length, batch_size] '''
  r_oracle_tgt = tf.expand_dims(oracle_tgt, axis=2)
  r_oracle_tgt = tf.tile(r_oracle_tgt, [1, 1, top_ks[-1]])
  imd_equal = tf.cast(predictions == r_oracle_tgt, int_type)
#   print("tf.shape(predictions):" + str(tf.shape(predictions)))
#   print("tf.shape(oracle_tgt):" + str(tf.shape(oracle_tgt)))
#   print("tf.shape(valid_mask):" + str(tf.shape(valid_mask)))
  token_accuracy = []
  for i in range(len(top_ks)):
    tpk_imd_equal = imd_equal * top_ks_tensors[i]
    imd_out_i = tf.cast(tf.reduce_sum(tpk_imd_equal, axis=2) >= 1, int_type)
    imd_out_i = imd_out_i * valid_mask
    imd_out_i_sum = tf.reduce_sum(imd_out_i)
    token_accuracy.append(tf.cast(imd_out_i_sum, float_type))
  ''' immediate_output shape: [seq_length, batch_size, len(top_ks)] '''
  
  ''' final output shape: [len(top_ks)] '''
  
  return token_accuracy



