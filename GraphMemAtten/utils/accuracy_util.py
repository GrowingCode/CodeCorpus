import tensorflow as tf
from meta_info.non_hyper_constant import float_type, int_type, top_ks,\
  top_ks_tensors
import numpy as np
from builtins import len


def compute_unit_expand_accuracy_of_sequences(unit_expand_base, unit_expand_start, unit_expand_end, raw_computed_en_seqs, raw_oracle_computed_en_seq, oracle_valid_mask, compute_one_whole=True):
#   print("tf.shape(raw_computed_en_seqs):" + str(tf.shape(raw_computed_en_seqs)))
#   print("tf.shape(raw_oracle_computed_en_seq):" + str(tf.shape(raw_oracle_computed_en_seq)))
#   print("tf.shape(oracle_valid_mask):" + str(tf.shape(oracle_valid_mask)))
  tf_seq_list = raw_computed_en_seqs.unstack()
  np_seq_list = [tf_seq.numpy() for tf_seq in tf_seq_list]
  l_size = len(np_seq_list)
  
  np_oracle_en_seq = raw_oracle_computed_en_seq.numpy()
  np_oracle_valid_mask = oracle_valid_mask.numpy()
  
  infer_size = np.size(np_seq_list[0])
  oracle_size = np.size(np_oracle_en_seq)
  r_size = min(infer_size, oracle_size)
  
  max_epos_right = 0
  max_whole_right = 0
  
  tpk_idx = 0
  f_each_acc = []
  f_whole_acc = []
  f_count = -1
  
  for i in range(l_size):
    nsl = np_seq_list[i]
    temp_pos_accurate_count = 0
    temp_pos_sub_unit_size = 0
    
    for j in range(r_size):
      if np_oracle_valid_mask[j]:
        infer_en = nsl[j]
        oracle_en = np_oracle_en_seq[j]
        infer_seq = get_unit_expand_sequence(unit_expand_base, unit_expand_start, unit_expand_end, infer_en)
        oracle_seq = get_unit_expand_sequence(unit_expand_base, unit_expand_start, unit_expand_end, oracle_en)
        pos_acc, pos_sub_unit_size = compare_two_sequences(infer_seq, oracle_seq)
        temp_pos_accurate_count += pos_acc
        temp_pos_sub_unit_size += pos_sub_unit_size
    
    if compute_one_whole:
      epos_right = temp_pos_accurate_count / temp_pos_sub_unit_size
      whole_right = float(temp_pos_accurate_count == temp_pos_sub_unit_size)
      f_count = 1
    else:
      epos_right = temp_pos_accurate_count
      whole_right = float(temp_pos_accurate_count == temp_pos_sub_unit_size) * temp_pos_sub_unit_size
      f_count = temp_pos_sub_unit_size
    
    max_epos_right = max(max_epos_right, epos_right)
    max_whole_right = max(max_whole_right, whole_right)
    
    assert tpk_idx < len(top_ks)
    if i == top_ks[tpk_idx]:
      f_each_acc.append(max_epos_right)
      f_whole_acc.append(max_whole_right)
      tpk_idx = tpk_idx + 1
  
  assert len(top_ks) == len(f_each_acc)
  assert len(top_ks) == len(f_whole_acc)
  
  return tf.convert_to_tensor(f_each_acc), tf.convert_to_tensor(f_whole_acc), tf.convert_to_tensor(f_count)


def get_unit_expand_sequence(unit_expand_base, unit_expand_start, unit_expand_end, en):
  en_start = unit_expand_start[en]
  en_end = unit_expand_end[en]
  seq = unit_expand_base[en_start:en_end+1]
  return seq


def compare_two_sequences(infer_seq, oracle_seq):
  size = np.size(oracle_seq)
  i_len = min(np.size(infer_seq), size)
  acc_count = 0
  for i in range(i_len):
    if infer_seq[i] == oracle_seq[i]:
      acc_count = acc_count + 1
  return acc_count, size


def compute_accuracy_of_sequences(raw_computed_en_seqs, raw_oracle_computed_en_seq, oracle_valid_mask, compute_one_whole=True):
#   print("tf.shape(raw_computed_en_seqs):" + str(tf.shape(raw_computed_en_seqs)))
#   print("tf.shape(raw_oracle_computed_en_seq):" + str(tf.shape(raw_oracle_computed_en_seq)))
#   print("tf.shape(oracle_valid_mask):" + str(tf.shape(oracle_valid_mask)))
  
  positive_idx = tf.where(oracle_valid_mask > 0)
#   print("tf.shape(tf.gather(raw_oracle_computed_en_seq, positive_idx)):" + str(tf.shape(tf.gather(raw_oracle_computed_en_seq, positive_idx))))
  oracle_computed_en_seq = tf.squeeze(tf.gather(raw_oracle_computed_en_seq, positive_idx), axis=1)
  computed_en_seqs = tf.squeeze(tf.gather(raw_computed_en_seqs, positive_idx, axis=1), axis=2)
  
#   print("tf.shape(oracle_computed_en_seq):" + str(tf.shape(oracle_computed_en_seq)))
#   print("tf.shape(computed_en_seqs):" + str(tf.shape(computed_en_seqs)))
#   print("oracle_computed_en_seq:" + str(oracle_computed_en_seq))
#   print("computed_en_seqs:" + str(computed_en_seqs))
  
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


def compute_batch_top_ks_accuracy(predictions, oracle_tgt, r_valid_mask):
  ''' predictions shape: [seq_length, batch_size, top_ks[-1]] '''
  ''' oracle_tgt shape: [seq_length, batch_size] '''
  ''' r_valid_mask shape: [seq_length, batch_size] '''
  r_oracle_tgt = tf.expand_dims(oracle_tgt, axis=2)
  r_oracle_tgt = tf.tile(r_oracle_tgt, [1, 1, top_ks[-1]])
  imd_equal = tf.cast(predictions == r_oracle_tgt, int_type)
#   print("tf.shape(predictions):" + str(tf.shape(predictions)))
#   print("tf.shape(oracle_tgt):" + str(tf.shape(oracle_tgt)))
#   print("tf.shape(valid_mask):" + str(tf.shape(valid_mask)))
  token_accuracy = tf.zeros([0], float_type)
  for i in range(len(top_ks)):
    tpk_imd_equal = imd_equal * top_ks_tensors[i]
    imd_out_i = tf.cast(tf.reduce_sum(tpk_imd_equal, axis=2) >= 1, int_type)
    imd_out_i = imd_out_i * r_valid_mask
    imd_out_i_sum = tf.reduce_sum(imd_out_i)
    token_accuracy = tf.concat([token_accuracy, [tf.cast(imd_out_i_sum, float_type)]], axis=0)
  ''' immediate_output shape: [seq_length, batch_size, len(top_ks)] '''
  
  ''' final output shape: [len(top_ks)] '''
  
  return token_accuracy


# def generate_token_type_filter_mask(token_type):
#   bool_mask = tf.logical_or(tf.equal(accuracy_filter_based_on_token_type, token_type), tf.equal(accuracy_filter_based_on_token_type, -1))
#   int_mask = tf.cast(bool_mask, int_type)
#   return int_mask


def generate_token_type_filter_valid_mask(valid_mask, token_type, accuracy_filter_token_type):
  bool_mask = tf.logical_or(tf.equal(accuracy_filter_token_type, token_type), tf.equal(accuracy_filter_token_type, -1))
  int_mask = tf.cast(bool_mask, int_type)
  r_mask = valid_mask * int_mask
  return r_mask


if __name__ == '__main__':
  ''' test1 '''
  l_base = tf.ones([1, 5], int_type)
  ll = []
  for i in range(10):
    ll.append(l_base + i)
  raw_computed_en_seqs = tf.concat(ll, axis=0)
  raw_oracle_computed_en_seq = tf.ones([5], int_type)
  oracle_valid_mask = tf.ones_like(raw_oracle_computed_en_seq)
  res1 = compute_accuracy_of_sequences(raw_computed_en_seqs, raw_oracle_computed_en_seq, oracle_valid_mask, compute_one_whole=True)
  print("str(res1):" + str(res1))
  ''' test2 '''
  predictions = tf.tile(tf.expand_dims(tf.expand_dims(tf.constant([0,1,2,3,4,5,6,7,8,9], int_type),0),0),[2,2,1])
  oracle_tgt = tf.constant([[1,2],[3,4]], int_type)
  r_valid_mask = tf.ones_like(oracle_tgt)
  res2 = compute_batch_top_ks_accuracy(predictions, oracle_tgt, r_valid_mask)
  print("str(res2):" + str(res2))









