import tensorflow as tf
from meta_info.non_hyper_constant import float_type, int_type, top_ks,\
  top_ks_tensors
import numpy as np
from builtins import len
from meta_info.hyper_parameter import lcs_accuracy_mode


# def compute_skt_unit_expand_accuracy_of_sequences(unit_expand_base, unit_expand_start, unit_expand_end, raw_computed_en_seqs, raw_oracle_computed_en_seq, oracle_valid_mask, compute_one_whole=True):
# #   print("tf.shape(raw_computed_en_seqs):" + str(tf.shape(raw_computed_en_seqs)))
# #   print("tf.shape(raw_oracle_computed_en_seq):" + str(tf.shape(raw_oracle_computed_en_seq)))
# #   print("tf.shape(oracle_valid_mask):" + str(tf.shape(oracle_valid_mask)))
#   tf_seq_list = tf.unstack(raw_computed_en_seqs)
#   np_seq_list = [tf_seq.numpy() for tf_seq in tf_seq_list]
#   l_size = len(np_seq_list)
#   
#   np_oracle_en_seq = raw_oracle_computed_en_seq.numpy()
#   np_oracle_valid_mask = oracle_valid_mask.numpy()
#   
#   infer_size = np.size(np_seq_list[0])
#   oracle_size = np.size(np_oracle_en_seq)
#   
#   max_epos_right = 0
#   max_whole_right = 0
#   
#   tpk_idx = 0
#   f_each_acc = [0 for _ in top_ks]
#   f_whole_acc = [0 for _ in top_ks]
#   f_count = 0
#   
#   oracle_unit_expand_seq_list = []
#   oracle_sub_unit_size = 0
#   
#   for k in range(oracle_size):
#     oracle_en = np_oracle_en_seq[k]
# #     if np_oracle_valid_mask[k]:
#     assert oracle_en > 2
#     oracle_seq = get_unit_expand_sequence(unit_expand_base, unit_expand_start, unit_expand_end, oracle_en)
#     os_size = np.size(oracle_seq)
#     oracle_unit_expand_seq = []
#     for i in range(os_size):
#       if oracle_seq[i] > 2:
#         oracle_unit_expand_seq.append(oracle_seq[i])
#         oracle_sub_unit_size += 1
#       else:
#         oracle_unit_expand_seq.append(None)
#     oracle_unit_expand_seq_list.extend(oracle_unit_expand_seq)
#     
# #     else:
# #       assert oracle_en <= 2
# #       oracle_unit_expand_seq_list.append(None)
#   r_size = min(infer_size, oracle_size)
#   
#   if oracle_sub_unit_size > 0:
#     if compute_one_whole:
#       f_count = 1
#     else:
#       f_count = oracle_sub_unit_size
#     
#     for i in range(l_size):
#       nsl = np_seq_list[i]
#       temp_pos_accurate_count = 0
#       
#       for j in range(r_size):
#         
#         if np_oracle_valid_mask[j]:
#           infer_en = nsl[j]
# #           assert infer_en > 2
#           if 2 < infer_en and infer_en < n_skt:
#             infer_seq = get_unit_expand_sequence(unit_expand_base, unit_expand_start, unit_expand_end, infer_en)
#             pos_acc = compare_two_sequences(infer_seq, oracle_unit_expand_seq_list[j])
#             temp_pos_accurate_count += pos_acc
#       
#       assert temp_pos_accurate_count <= oracle_sub_unit_size
#       
#       if compute_one_whole:
#         epos_right = temp_pos_accurate_count / oracle_sub_unit_size
#         whole_right = float(temp_pos_accurate_count == oracle_sub_unit_size)
#       else:
#         epos_right = temp_pos_accurate_count
#         whole_right = float(temp_pos_accurate_count == oracle_sub_unit_size) * oracle_sub_unit_size
#       
#       max_epos_right = max(max_epos_right, epos_right)
#       max_whole_right = max(max_whole_right, whole_right)
#       
#       assert tpk_idx < len(top_ks)
#       if i+1 == top_ks[tpk_idx]:
#         f_each_acc[tpk_idx] += max_epos_right
#         f_whole_acc[tpk_idx] += max_whole_right
#         tpk_idx = tpk_idx + 1
#     
# #   assert len(top_ks) == len(f_each_acc)
# #   assert len(top_ks) == len(f_whole_acc)
#   
#   return tf.convert_to_tensor(f_each_acc, float_type), tf.convert_to_tensor(f_whole_acc, float_type), tf.convert_to_tensor(f_count, int_type)


def lcs(a,b):
  lena=len(a)
  lenb=len(b)
  c=[[0 for _ in range(lenb+1)] for _ in range(lena+1)]
  flag=[[0 for _ in range(lenb+1)] for _ in range(lena+1)]
  for i in range(lena):
    for j in range(lenb):
      if a[i]==b[j]:
        c[i+1][j+1]=c[i][j]+1
        flag[i+1][j+1]='ok'
      elif c[i+1][j]>c[i][j+1]:
        c[i+1][j+1]=c[i+1][j]
        flag[i+1][j+1]='left'
      else:
        c[i+1][j+1]=c[i][j+1]
        flag[i+1][j+1]='up'
  return c,flag, c[-1][-1]


def print_lcs(flag,a,i,j):
  if i==0 or j==0:
    return
  if flag[i][j]=='ok':
    print_lcs(flag,a,i-1,j-1)
    print(a[i-1],end='')
  elif flag[i][j]=='left':
    print_lcs(flag,a,i,j-1)
  else:
    print_lcs(flag,a,i-1,j)


# def compare_two_sequences(infer_seq, oracle_seq):
#   size = np.size(oracle_seq)
#   i_len = min(np.size(infer_seq), size)
#   acc_count = 0
#   if lcs_accuracy_mode:
#     _, _, acc_count = lcs(infer_seq, oracle_seq)
#   else:
#     for i in range(i_len):
#       if infer_seq[i] == oracle_seq[i]:
#         acc_count = acc_count + 1
#   return acc_count

# oracle_valid_mask, 

def flatten_nested_lists(seq):
  res = []
  for ele in seq:
    if isinstance(ele, list):
      res.extend(ele)
    else:
      res.append(ele)
  return res

def compute_accuracy_of_sequences(raw_computed_en_seqs, raw_oracle_computed_en_seq, compute_one_whole=True):
#   print("tf.shape(raw_computed_en_seqs):" + str(tf.shape(raw_computed_en_seqs)))
#   print("tf.shape(raw_oracle_computed_en_seq):" + str(tf.shape(raw_oracle_computed_en_seq)))
#   print("tf.shape(oracle_valid_mask):" + str(tf.shape(oracle_valid_mask)))
#   tf_seq_list = tf.unstack(raw_computed_en_seqs)
#   np_seq_list = [tf_seq.numpy() for tf_seq in tf_seq_list]
  l_size = len(raw_computed_en_seqs)
  
  if lcs_accuracy_mode:
    np_seq_list = []
    for i in range(l_size):
      nsl = raw_computed_en_seqs[i]
      np_seq_list.append(flatten_nested_lists(nsl))
    np_oracle_en_seq = flatten_nested_lists(raw_oracle_computed_en_seq)
  else:
    np_seq_list = raw_computed_en_seqs
    np_oracle_en_seq = raw_oracle_computed_en_seq
  
#   np_oracle_en_seq = raw_oracle_computed_en_seq.numpy()
#   np_oracle_valid_mask = oracle_valid_mask.numpy()
#   np_oracle_seq_valid_number = oracle_seq_valid_number.numpy()
  
  oracle_size = np.size(np_oracle_en_seq)
  
  max_epos_right = 0
  max_whole_right = 0
  
  tpk_idx = 0
  f_each_acc = [0 for _ in top_ks]
  f_whole_acc = [0 for _ in top_ks]
  f_count = 0
  
#   oracle_unit_expand_seq_list = []
  oracle_sub_unit_size = 0
  
  for k in range(oracle_size):
    oracle_en = np_oracle_en_seq[k]
#     if np_oracle_valid_mask[k]:
    if oracle_en != None:
#       assert oracle_en > 2 + n_base
#       oracle_unit_expand_seq_list.append(oracle_en)
      if isinstance(oracle_en, list):
        for t in oracle_en:
          if t != None:
            oracle_sub_unit_size += 1
      else:
        oracle_sub_unit_size += 1
      
#     else:
#       assert n_base <= oracle_en and oracle_en <= 2 + n_base, "wrong oracle_en:" + str(oracle_en) + "#n_base:" + str(n_base)
#       oracle_unit_expand_seq_list.append(None)
#   assert np_oracle_seq_valid_number == oracle_sub_unit_size
  
  if oracle_sub_unit_size > 0:
    if compute_one_whole:
      f_count = 1
    else:
      f_count = oracle_sub_unit_size
    
    for i in range(l_size):
      nsl = np_seq_list[i]
      
      infer_size = np.size(nsl)
      r_size = min(infer_size, oracle_size)
      
      if lcs_accuracy_mode:
        _, _, temp_pos_accurate_count = lcs(nsl, np_oracle_en_seq)
      else:
        temp_pos_accurate_count = 0
        for j in range(r_size):
          nsl_j = nsl[j]
          np_oracle_en_seq_j = np_oracle_en_seq[j]
          if isinstance(nsl_j, list):
            assert isinstance(np_oracle_en_seq_j, list)
            nsl_j_size = len(nsl_j)
            np_oracle_en_seq_j_size = len(np_oracle_en_seq_j)
            j_size = min(nsl_j_size, np_oracle_en_seq_j_size)
            for k in range(j_size):
              if nsl_j[k] == np_oracle_en_seq_j[k]:
                temp_pos_accurate_count = temp_pos_accurate_count + 1
          else:
            if nsl[j] == np_oracle_en_seq[j]:
              temp_pos_accurate_count = temp_pos_accurate_count + 1
      
#       print("temp_pos_accurate_count1:" + str(temp_pos_accurate_count1) + "#temp_pos_accurate_count2:" + str(temp_pos_accurate_count))
#       for j in range(r_size):
#         oracle_en = np_oracle_en_seq[k]
# #         if np_oracle_valid_mask[j]:
#         if oracle_en != None:
#           infer_en = nsl[j]
# #           assert infer_en > 2
# #           if infer_en < n_skt:
#           pos_acc = (1 if infer_en == oracle_en else 0)
#           temp_pos_accurate_count += pos_acc
      
      assert temp_pos_accurate_count <= oracle_sub_unit_size
      
      if compute_one_whole:
        epos_right = temp_pos_accurate_count / oracle_sub_unit_size
        whole_right = float(temp_pos_accurate_count == oracle_sub_unit_size)
      else:
        epos_right = temp_pos_accurate_count
        whole_right = float(temp_pos_accurate_count == oracle_sub_unit_size) * oracle_sub_unit_size
      
      max_epos_right = max(max_epos_right, epos_right)
      max_whole_right = max(max_whole_right, whole_right)
      
      assert tpk_idx < len(top_ks)
      if i+1 == top_ks[tpk_idx]:
        f_each_acc[tpk_idx] += max_epos_right
        f_whole_acc[tpk_idx] += max_whole_right
        tpk_idx = tpk_idx + 1
    
#   assert len(top_ks) == len(f_each_acc)
#   assert len(top_ks) == len(f_whole_acc)
#   print("f_each_acc:" + str(f_each_acc))
#   print("f_count:" + str(f_count))
  return tf.convert_to_tensor(f_each_acc, float_type), tf.convert_to_tensor(f_whole_acc, float_type), tf.convert_to_tensor(f_count, int_type)
#   
#   positive_idx = tf.where(oracle_valid_mask > 0)
# #   print("tf.shape(tf.gather(raw_oracle_computed_en_seq, positive_idx)):" + str(tf.shape(tf.gather(raw_oracle_computed_en_seq, positive_idx))))
#   oracle_computed_en_seq = tf.squeeze(tf.gather(raw_oracle_computed_en_seq, positive_idx), axis=1)
#   computed_en_seqs = tf.squeeze(tf.gather(raw_computed_en_seqs, positive_idx, axis=1), axis=2)
#   
# #   print("tf.shape(oracle_computed_en_seq):" + str(tf.shape(oracle_computed_en_seq)))
# #   print("tf.shape(computed_en_seqs):" + str(tf.shape(computed_en_seqs)))
# #   print("oracle_computed_en_seq:" + str(oracle_computed_en_seq))
# #   print("computed_en_seqs:" + str(computed_en_seqs))
#   
#   e_lens = tf.ones_like(oracle_computed_en_seq)
#   eq_all_lens_int = tf.reduce_sum(e_lens)
#   e_lens = tf.cast(e_lens, float_type)
#   eq_all_lens = tf.reduce_sum(e_lens)
#   
#   def compute_acc_cond(i, i_len, *_):
#     return i < i_len
#   
#   def compute_acc_body(i, i_len, epos_acc, whole_acc):
#     one_computed_en_seq = computed_en_seqs[i]
#     tc = tf.zeros([tf.shape(oracle_computed_en_seq)[-1] - tf.shape(one_computed_en_seq)[-1]], int_type) - 1
#     r_one_computed_en_seq = tf.concat([one_computed_en_seq, tc], axis=0)
#     
#     eq = tf.cast(tf.equal(r_one_computed_en_seq, oracle_computed_en_seq), float_type)
#     eq_lens = tf.reduce_sum(eq * e_lens)
# #     eq_all_acc = tf.reduce_sum(eq)
# #     eq_all_count = tf.cast(tf.shape(eq)[-1], float_type)
#     if compute_one_whole:
#       epos_right = eq_lens / eq_all_lens
#       whole_right = tf.cast(tf.equal(eq_lens, eq_all_lens), float_type)
#     else:
#       epos_right = eq_lens
#       whole_right = tf.cast(tf.equal(eq_lens, eq_all_lens), float_type) * eq_all_lens
#     
#     epos_r_acc = tf.maximum(epos_acc[-1], epos_right)
#     whole_r_acc = tf.maximum(whole_acc[-1], whole_right)
#     
#     epos_acc = tf.concat([epos_acc, [epos_r_acc]], axis=0)
#     whole_acc = tf.concat([whole_acc, [whole_r_acc]], axis=0)
#     
#     return i+1, i_len, epos_acc, whole_acc
#   
#   n = tf.shape(computed_en_seqs)[0]
#   each_acc = tf.zeros([1], float_type)
#   whole_acc = tf.zeros([1], float_type)
#   _, _, each_acc, whole_acc = tf.while_loop(compute_acc_cond, compute_acc_body, [0, n, each_acc, whole_acc], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None]), tf.TensorShape([None])])
#   
#   f_each_acc = tf.zeros([0], float_type)
#   f_whole_acc = tf.zeros([0], float_type)
#   acc_len = tf.shape(each_acc)[-1]
#   tpk_len = len(top_ks)
#   for i in range(tpk_len):
#     tpk = top_ks[i]
#     r_sel = tf.minimum(tpk, acc_len-1)
#     f_each_acc = tf.concat([f_each_acc, [each_acc[r_sel]]], axis=0)
#     f_whole_acc = tf.concat([f_whole_acc, [whole_acc[r_sel]]], axis=0)
#   
#   if compute_one_whole:
#     f_count = tf.constant(1, int_type)
#   else:
#     f_count = eq_all_lens_int
#   
#   return f_each_acc, f_whole_acc, f_count


def compute_batch_top_ks_accuracy(predictions, oracle_tgt, r_valid_mask, r_token_type):
  ''' predictions shape: [seq_length, batch_size, top_ks[-1]] '''
  ''' oracle_tgt shape: [seq_length, batch_size] '''
  ''' r_valid_mask shape: [seq_length, batch_size] '''
  r_oracle_tgt = tf.expand_dims(oracle_tgt, axis=2)
  r_oracle_tgt = tf.tile(r_oracle_tgt, [1, 1, top_ks[-1]])
  imd_equal = tf.cast(predictions == r_oracle_tgt, int_type)
#   print("tf.shape(predictions):" + str(tf.shape(predictions)))
#   print("tf.shape(oracle_tgt):" + str(tf.shape(oracle_tgt)))
#   print("tf.shape(valid_mask):" + str(tf.shape(valid_mask)))
  all_token_accuracy = tf.zeros([0], float_type)
  t0_token_accuracy = tf.zeros([0], float_type)
  t1_token_accuracy = tf.zeros([0], float_type)
  
  t0_mask = r_valid_mask * tf.cast(r_token_type == 0, int_type)
  t1_mask = r_valid_mask * tf.cast(r_token_type == 1, int_type)
  
  for i in range(len(top_ks)):
    tpk_imd_equal = imd_equal * top_ks_tensors[i]
    imd_out_i = tf.cast(tf.reduce_sum(tpk_imd_equal, axis=2) >= 1, int_type)
    ''' immediate_output shape: [seq_length, batch_size] '''# , len(top_ks)
    all_imd_out_i = imd_out_i * r_valid_mask
    all_imd_out_i_sum = tf.reduce_sum(all_imd_out_i)
    all_token_accuracy = tf.concat([all_token_accuracy, [tf.cast(all_imd_out_i_sum, float_type)]], axis=0)
    
    t0_imd_out_i = imd_out_i * t0_mask
    t0_imd_out_i_sum = tf.reduce_sum(t0_imd_out_i)
    t0_token_accuracy = tf.concat([t0_token_accuracy, [tf.cast(t0_imd_out_i_sum, float_type)]], axis=0)
    
    t1_imd_out_i = imd_out_i * t1_mask
    t1_imd_out_i_sum = tf.reduce_sum(t1_imd_out_i)
    t1_token_accuracy = tf.concat([t1_token_accuracy, [tf.cast(t1_imd_out_i_sum, float_type)]], axis=0)
  
  ''' final output shape: [len(top_ks)] '''
  all_token_count = tf.reduce_sum(r_valid_mask)
  t0_token_count = tf.reduce_sum(t0_mask)
  t1_token_count = tf.reduce_sum(t1_mask)
  return all_token_accuracy, all_token_count, t0_token_accuracy, t0_token_count, t1_token_accuracy, t1_token_count


# def generate_token_type_filter_mask(token_type):
#   bool_mask = tf.logical_or(tf.equal(accuracy_filter_based_on_token_type, token_type), tf.equal(accuracy_filter_based_on_token_type, -1))
#   int_mask = tf.cast(bool_mask, int_type)
#   return int_mask


def generate_token_type_filter_valid_mask(valid_mask, token_type, accuracy_filter_token_type):
  bool_mask = tf.logical_or(tf.equal(accuracy_filter_token_type, token_type), tf.equal(accuracy_filter_token_type, -1))
  int_mask = tf.cast(bool_mask, int_type)
  r_mask = valid_mask * int_mask
  return r_mask


def test_lcs():
  a='ABCBDAB'
  b='BDCABA'
  print(a)
  print(b)
  print('====')
  c,flag,count=lcs(a,b)
  print_lcs(flag,a,len(a),len(b))
  print('')
  print(count)
  assert c != None
#   for i in c:
#     print(i)
#   print('')
#   for j in flag:
#     print(j)
#   print('')
#   print('')
  


if __name__ == '__main__':
  test_lcs()
#   ''' test1 '''
#   l_base = tf.ones([1, 5], int_type)
#   ll = []
#   for i in range(10):
#     ll.append(l_base + i)
#   raw_computed_en_seqs = tf.concat(ll, axis=0)
#   raw_oracle_computed_en_seq = tf.ones([5], int_type)
#   oracle_valid_mask = tf.ones_like(raw_oracle_computed_en_seq)
#   res1 = compute_accuracy_of_sequences(raw_computed_en_seqs, raw_oracle_computed_en_seq, oracle_valid_mask, compute_one_whole=True)
#   print("str(res1):" + str(res1))
#   ''' test2 '''
#   predictions = tf.tile(tf.expand_dims(tf.expand_dims(tf.constant([0,1,2,3,4,5,6,7,8,9], int_type),0),0),[2,2,1])
#   oracle_tgt = tf.constant([[1,2],[3,4]], int_type)
#   r_valid_mask = tf.ones_like(oracle_tgt)
#   res2 = compute_batch_top_ks_accuracy(predictions, oracle_tgt, r_valid_mask)
#   print("str(res2):" + str(res2))









