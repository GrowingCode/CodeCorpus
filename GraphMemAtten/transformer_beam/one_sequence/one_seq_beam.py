from meta_info.hyper_parameter import oracle_mem_len, \
  oracle_tgt_len, accuracy_based_on_whole, \
  memory_train_test_beam_consistent
from meta_info.non_hyper_constant import int_type, float_type, \
  standard_infer_test, multi_infer_test, debug_in_test_beam, top_ks,\
  stand_infer_skt_ens, stand_oracle_skt_ens, multi_infer_token_ens,\
  multi_oracle_token_ens, stand_infer_token_ens, stand_oracle_token_ens,\
  multi_infer_skt_ens, multi_oracle_skt_ens
import numpy as np
import tensorflow as tf
from transformer_beam.one_sequence.one_step.one_step_infer import OneStepMultiInfer, \
  OneStepStandInfer, framework_skt_infer, framework_token_infer
from utils.accuracy_util import compute_accuracy_of_sequences
from utils.memory_util import get_recent_fixed_length_memory, \
  update_recent_fixed_length_memory, get_specified_varied_length_memory
from utils.unit_expand_util import get_unit_expand_sequence, \
  get_unit_expand_sequence_list, replace_unk_with_none_in_list


class OneSeqBeam():
  
  def __init__(self, transformer_model, multi_decode_model):
    self.transformer_model = transformer_model
    self.multi_decode_model = multi_decode_model
    # multi_position_transfer 
  
  def __call__(self, mems, whole_seq, valid_mask, parent_hint, position_hint, part_seq_skip, token_type, whole_seq_exact, decode_mode):
    origin_mems_len = tf.shape(mems[0])[0]
    
    ''' generate stored memory for whole sequence '''
    seq_len = tf.shape(whole_seq)[0]
    
    def mem_gen_cond(idx, *_):
      return tf.less(idx, seq_len)
    
    def mem_gen_body(idx, *all_mems):
      all_mems = list(all_mems)
      seq_part_size = tf.minimum(idx + oracle_tgt_len, seq_len) - idx
      dec_inp = tf.slice(whole_seq, [idx], [seq_part_size])
      dec_inp = tf.expand_dims(dec_inp, axis=1)
      target = tf.zeros_like(dec_inp) - 1
      
      temp_valid_mask = tf.slice(valid_mask, [idx], [seq_part_size])
      temp_valid_mask = tf.expand_dims(temp_valid_mask, axis=1)
      
      temp_token_type = tf.slice(token_type, [idx], [seq_part_size])
      temp_token_type = tf.expand_dims(temp_token_type, axis=1)
      
      temp_parent_hint = tf.slice(parent_hint, [idx], [seq_part_size])
      temp_parent_hint = tf.expand_dims(temp_parent_hint, axis=1)
      
      temp_position_hint = tf.slice(position_hint, [idx], [seq_part_size])
      temp_position_hint = tf.expand_dims(temp_position_hint, axis=1)
      
      temp_mems = get_recent_fixed_length_memory(all_mems, oracle_mem_len)
      
      _, _, predictions, _, new_mems = self.transformer_model.transformer(dec_inp, target, temp_mems, temp_valid_mask, temp_token_type, temp_parent_hint, temp_position_hint, is_training=0)#, mem_len=oracle_mem_len
      if debug_in_test_beam:
        print("dec_inp:" + str(tf.squeeze(dec_inp).numpy()) + "#predictions:" + str(tf.squeeze(predictions).numpy()) + "#token_type:" + str(tf.squeeze(token_type).numpy()) + "#valid_mask:" + str(tf.squeeze(valid_mask).numpy()))
      
      new_all_mems = update_recent_fixed_length_memory(all_mems, new_mems)
      
      return (idx + oracle_tgt_len, *new_all_mems)
    
    with tf.device('/gpu:0'):
      _, *all_mems = tf.while_loop(mem_gen_cond, mem_gen_body, [tf.constant(0, int_type), *mems], parallel_iterations=1)
    
    def osb_cond(i, *_):
      i_valid = tf.cast(tf.less(i, seq_len), int_type)
      r_i = tf.stack([tf.constant(0, int_type), i])[i_valid]
      return tf.greater(part_seq_skip[r_i], tf.constant(0, int_type))
  
    # skt_each_acc, skt_whole_acc, skt_count, 
    def osb_body(i, skt_each_acc, skt_whole_acc, skt_count, token_each_acc, token_whole_acc, token_count):
#       o_skt_start = skt_token_split[i][0]
#       o_skt_end = skt_token_split[i][1]
#       o_token_end = skt_token_split[i][2]
      
#       skt_last_before = o_skt_start - 1
#       skt_last_before_valid = tf.cast(tf.greater_equal(skt_last_before, 0), int_type)
#       skt_real_last_before = tf.stack([last_token_before_whole_seq, skt_last_before])[skt_last_before_valid]
#       skt_part_seq = tf.slice(whole_seq, [o_skt_start, 0], [o_skt_end-o_skt_start+1, -1])
#       
#       skt_mems_end = origin_mems_len + skt_last_before
#       skt_mems_start = tf.maximum(skt_mems_end - oracle_predict_mem_len, 0)
#       skt_mems_before_last = []
#       for i in range(n_layer):
#         skt_mems_before_last.append(tf.slice(all_mems, [skt_mems_start, 0, 0], [skt_mems_end-skt_mems_start+1, -1, -1]))
#       skt_f_each_acc, skt_f_whole_acc, skt_f_count = self.infer_and_compute_accuracy(skt_mems_before_last, skt_real_last_before, skt_part_seq, beam_mode)
#       skt_each_acc += skt_f_each_acc
#       skt_whole_acc += skt_f_whole_acc
#       skt_count += skt_f_count
      
      token_last_before_index = i-1
#       token_last_before_valid = tf.cast(tf.greater_equal(token_last_before, 0), int_type)
#       token_real_last_before = tf.stack([last_token_before_whole_seq, token_last_before])[token_last_before_valid]
      token_part_seq = tf.slice(whole_seq, [i], [part_seq_skip[i]])
#       token_part_valid_mask = tf.slice(valid_mask, [i], [part_seq_skip[i]])
      token_part_parent_hint = tf.slice(parent_hint, [i], [part_seq_skip[i]])
      token_part_position_hint = tf.slice(position_hint, [i], [part_seq_skip[i]])
      token_part_token_type = tf.slice(token_type, [i], [part_seq_skip[i]])
      token_part_seq_exact = tf.slice(whole_seq_exact, [i], [part_seq_skip[i]])
      
#       token_mems_end = token_last_before_index - 1 + origin_mems_len
#       token_mems_start = tf.maximum(token_mems_end - oracle_test_mem_len + 1, 0)
# #       print("token_mems_start:" + str(token_mems_start) + "#token_mems_end:" + str(token_mems_end))
#       token_mems_before_last = []
#       for j in range(n_layer):
#         token_mems_before_last.append(tf.slice(all_mems[j], [token_mems_start, 0, 0], [token_mems_end-token_mems_start+1, -1, -1]))
      token_mems_before_last = get_specified_varied_length_memory(all_mems, origin_mems_len + token_last_before_index - 1, oracle_mem_len, memory_train_test_beam_consistent)
      
      if debug_in_test_beam:
        print("tf.shape(token_mems_before_last):" + str(tf.shape(token_mems_before_last)))
      
      token_last_before = whole_seq[token_last_before_index]
      r_token_last_before = tf.expand_dims(tf.expand_dims(token_last_before, axis=0), axis=1)
#       print("r_token_last_before:" + str(r_token_last_before))
      skt_f_each_acc, skt_f_whole_acc, skt_f_count, token_f_each_acc, token_f_whole_acc, token_f_count = self.infer_and_compute_accuracy(token_mems_before_last, r_token_last_before, token_part_seq, token_part_token_type, token_part_parent_hint, token_part_position_hint, token_part_token_type, token_part_seq_exact, decode_mode)
      skt_each_acc += skt_f_each_acc
      skt_whole_acc += skt_f_whole_acc
      skt_count += skt_f_count
      token_each_acc += token_f_each_acc
      token_whole_acc += token_f_whole_acc
      token_count += token_f_count
      
      # skt_each_acc, skt_whole_acc, skt_count, 
      return i+part_seq_skip[i], skt_each_acc, skt_whole_acc, skt_count, token_each_acc, token_whole_acc, token_count
    
    i = tf.constant(1, int_type)
#     skt_each_acc, skt_whole_acc, skt_count = tf.constant(0, float_type), tf.constant(0, float_type), tf.constant(0, int_type)
    skt_each_acc, skt_whole_acc, skt_count = tf.constant(0, float_type), tf.constant(0, float_type), tf.constant(0, int_type)
    token_each_acc, token_whole_acc, token_count = tf.constant(0, float_type), tf.constant(0, float_type), tf.constant(0, int_type)
    # skt_each_acc, skt_whole_acc, skt_count, 
    # i_len, skt_each_acc, skt_whole_acc, skt_count, 
    _, skt_each_acc, skt_whole_acc, skt_count, token_each_acc, token_whole_acc, token_count = tf.while_loop(osb_cond, osb_body, [i, skt_each_acc, skt_whole_acc, skt_count, token_each_acc, token_whole_acc, token_count], parallel_iterations=1)
    # skt_each_acc, skt_whole_acc, skt_count, 
    return skt_each_acc, skt_whole_acc, skt_count, token_each_acc, token_whole_acc, token_count, all_mems
  
  # part_valid_mask, 
  def infer_and_compute_accuracy(self, mems_before_last, last_token_before_part_seq, part_seq, part_type_hint, part_parent_hint, part_position_hint, part_token_type, part_seq_exact, decode_mode):
    ptt_np = part_token_type.numpy()
    e0 = np.all(ptt_np == 0)
    e1 = np.all(ptt_np == 1)
    assert e0 or e1
    
    part_seq_len = tf.shape(part_seq)[0]
    part_seq_exact_len = tf.shape(part_seq_exact)[0]
    assert part_seq_len == part_seq_exact_len
#     print("part_seq_exact.numpy():" + str(part_seq_exact.numpy()))
#     r_part_seq_exact = get_unit_expand_sequence(part_seq_exact.numpy().tolist(), -1)
#     r_part_seq_len = len(r_part_seq_exact)
    
    predict_len = part_seq_len
    final_trim_len = -1
#     if beam_infer_with_skt_e_full_length:
#       predict_len = r_part_seq_len
#       final_trim_len = r_part_seq_len
#     print("r_part_seq_exact:" + str(r_part_seq_exact))
#     with tf.device('/gpu:0'):
    if decode_mode == standard_infer_test:
      inferrer = OneStepStandInfer(self.transformer_model, mems_before_last, last_token_before_part_seq)
    elif decode_mode == multi_infer_test:
      inferrer = OneStepMultiInfer(self.transformer_model, self.multi_decode_model, mems_before_last, last_token_before_part_seq)
    else:
      assert False
#     print("inferred_ens.numpy():" + str(inferred_ens.numpy()))
    if e0:
      inferred_ens = framework_skt_infer(inferrer, part_type_hint, part_parent_hint, part_position_hint, predict_len)
#       if skeleton_mode == skeleton_e:
#         skt_f_each_acc, skt_f_whole_acc, skt_f_count = compute_accuracy_of_sequences(inferred_ens, part_seq_exact, part_valid_mask, compute_one_whole=accuracy_based_on_whole)
#       elif skeleton_mode == skeleton_pe:
#         skt_f_each_acc, skt_f_whole_acc, skt_f_count = compute_skt_unit_expand_accuracy_of_sequences(all_skt_pe_to_each_base, all_skt_pe_to_each_start, all_skt_pe_to_each_end, inferred_ens, part_seq_exact, part_valid_mask, compute_one_whole=accuracy_based_on_whole)
#       elif skeleton_mode == skeleton_one:
#         skt_f_each_acc, skt_f_whole_acc, skt_f_count = compute_skt_unit_expand_accuracy_of_sequences(all_skt_one_to_each_base, all_skt_one_to_each_start, all_skt_one_to_each_end, inferred_ens, part_seq_exact, part_valid_mask, compute_one_whole=accuracy_based_on_whole)
#       else:
#         assert False
#       r_part_seq_len = len(r_part_seq_exact)
      r_inferred_ens = get_unit_expand_sequence_list(inferred_ens, final_trim_len)
      r_part_seq_exact = replace_unk_with_none_in_list(get_unit_expand_sequence(part_seq_exact.numpy().tolist(), -1))
#       print("r_inferred_ens:" + str(r_inferred_ens))
      skt_f_each_acc, skt_f_whole_acc, skt_f_count = compute_accuracy_of_sequences(r_inferred_ens, r_part_seq_exact, compute_one_whole=accuracy_based_on_whole)
      token_f_each_acc, token_f_whole_acc, token_f_count = tf.zeros([len(top_ks)], float_type), tf.zeros([len(top_ks)], float_type), tf.constant(0, int_type)
    elif e1:
      r_inferred_ens = framework_token_infer(inferrer, part_type_hint, part_parent_hint, part_position_hint, predict_len)
      skt_f_each_acc, skt_f_whole_acc, skt_f_count = tf.zeros([len(top_ks)], float_type), tf.zeros([len(top_ks)], float_type), tf.constant(0, int_type)
      r_part_seq_exact = replace_unk_with_none_in_list(part_seq_exact.numpy().tolist())
      token_f_each_acc, token_f_whole_acc, token_f_count = compute_accuracy_of_sequences(r_inferred_ens, r_part_seq_exact, compute_one_whole=accuracy_based_on_whole)
    else:
      assert False
    
    if decode_mode == standard_infer_test:
      if e0:
        stand_infer_skt_ens.append(r_inferred_ens)
        stand_oracle_skt_ens.append(r_part_seq_exact)
      elif e1:
        stand_infer_token_ens.append(r_inferred_ens)
        stand_oracle_token_ens.append(r_part_seq_exact)
      else:
        assert False
    elif decode_mode == multi_infer_test:
      if e0:
        multi_infer_skt_ens.append(r_inferred_ens)
        multi_oracle_skt_ens.append(r_part_seq_exact)
      elif e1:
        multi_infer_token_ens.append(r_inferred_ens)
        multi_oracle_token_ens.append(r_part_seq_exact)
    else:
      assert False
#     print("inferred_ens:" + str(inferred_ens) + "#last_token_before_part_seq:" + str(last_token_before_part_seq) + "#part_seq:" + str(part_seq))
    return skt_f_each_acc, skt_f_whole_acc, skt_f_count, token_f_each_acc, token_f_whole_acc, token_f_count
  
#   def infer(self, mems_before_last, last_token, steps, initial_batch_size_is_one=True, initial_probs=None, initial_ens=None):
#     ''' if the following 1 means batch_size, then it may be top_ks[-1] if initial_batch_size_is_one==False '''
#     ''' last_token shape: [1, 1] (or [1, batch_size]) '''
#     ''' here mems_before_last shape must be [n_layer memory_length 1 feature_size] '''
#     ''' here mems_before_last shape should be extended to [n_layer memory_length batch_size feature_size] '''
#     r_steps = steps
# #     tf.cast(tf.minimum(multi_infer_num, steps), int_type)
#     
#     def infer_cond(i, *_):
#       return tf.less(i, r_steps)
#     
#     def infer_body(i, probs, ens, l_token, *mems_tuple):
#       ''' in the following, batch_size should be top_ks[-1]]] '''
#       ''' probs shape should be [batch_size] '''
#       ''' ens shape should be [batch_size, varied_with_max_length_(steps + 1 or >> 1) ] '''
#       ''' l_token shape should be [1, batch_size] '''
#       mems = list(mems_tuple)
#       ''' mems shape should be [n_layer memory_length batch_size feature_size] '''
#       _, r_probs, r_predictions, _, new_mems = self.transformer_model.transformer(l_token, tf.zeros_like(l_token)-1, mems, tf.ones_like(l_token), is_training=False, mem_len=oracle_mem_len)
#       new_mems = update_recent_fixed_length_memory(mems, new_mems)
#       if additional_filter_memory_when_beam_step_inferring:
#         new_mems = get_specified_varied_length_memory(new_mems, -1, oracle_mem_len, memory_train_test_beam_consistent)
# #       print("r_predictions in infer_body:" + str(r_predictions))
#       ''' probs          should be [1, batch_size, top_ks[-1]] '''
#       ''' predictions    should be [1, batch_size, top_ks[-1]] '''
#       r_probs = tf.squeeze(r_probs, [0])
#       r_predictions = tf.squeeze(r_predictions, [0])
#       ''' r_predictions should be [batch_size, top_ks[-1]] '''
#       
#       new_probs = batch_cartesian_add_each_scalar_in_vect(probs, r_probs)
#       new_ens = batch_cartesian_concat_one_dim_vect_and_each_scalar_in_vect(ens, r_predictions)
# #       print("tf.shape(new_ens):" + str(tf.shape(new_ens)))
#       ''' new_probs should be [batch_size * top_ks[-1]] '''
#       ''' new_ens should be [batch_size * top_ks[-1], varied_with_max_length_steps + 1] '''
#       
#       probs_values, probs_indices = tf.nn.top_k(new_probs, top_ks[-1])
#       ens_values = tf.gather(new_ens, probs_indices)
#       
#       ''' update memory '''
#       mems_indices = tf.math.floordiv(probs_indices, top_ks[-1])
#       for j in range(n_layer):
#         new_mems[j] = tf.gather(new_mems[j], mems_indices, axis=1)
#       
#       last_col_ens = tf.expand_dims(ens_values[:, -1], axis=0)
#       return (i+1, probs_values, ens_values, last_col_ens, *new_mems)
#     
#     if not initial_batch_size_is_one:
#       i = tf.constant(0, int_type)
#       probs = initial_probs# tf.zeros([top_ks[-1]], float_type)
#       ens = initial_ens# tf.zeros([top_ks[-1], 0], int_type)
#       l_token = last_token
#       wloop_mems = mems_before_last
# #       print(tf.shape(wloop_mems))
#     else:
#       _, r_probs, r_predictions, _, new_mems = self.transformer_model.transformer(last_token, tf.zeros_like(last_token)-1, mems_before_last, tf.ones_like(last_token), is_training=False, mem_len=oracle_mem_len)
#       new_mems = update_recent_fixed_length_memory(mems_before_last, new_mems)
#       if additional_filter_memory_when_beam_step_inferring:
#         new_mems = get_specified_varied_length_memory(new_mems, -1, oracle_mem_len, memory_train_test_beam_consistent)
# #       print("r_predictions out infer_body:" + str(r_predictions))
#       i = tf.constant(1, int_type)
#       probs = tf.squeeze(r_probs, axis=[0, 1])
#       raw_ens = tf.squeeze(r_predictions, axis=[0, 1])
#       ens = tf.expand_dims(raw_ens, axis=1)
#       l_token = tf.expand_dims(raw_ens, axis=0)
# #     i_len = tf.constant(steps, int_type)
# #     probs = tf.zeros([top_ks[-1]], float_type)
# #     ens = tf.zeros([top_ks[-1], 1], int_type) - 1
#       wloop_mems = []
#       for j in range(n_layer):
#         wloop_mems.append(tf.tile(new_mems[j], [1, top_ks[-1], 1]))
# #     last_token = tf.tile(last_token, [1, top_ks[-1]])
#     mems_shapes = get_varied_memory_shape_in_while_loop()
# #     i_len, 
#     _, probs, ens, *_ = tf.while_loop(infer_cond, infer_body, loop_vars=[i, probs, ens, l_token, *wloop_mems], shape_invariants=[tf.TensorShape(()), tf.TensorShape([top_ks[-1]]), tf.TensorShape([top_ks[-1], None]), tf.TensorShape([1, top_ks[-1]]), *mems_shapes], parallel_iterations=1)
#     
# #     print("steps:" + str(steps))
# #     print("tf.shape(ens):" + str(tf.shape(ens)))
#     
# #     ens = tf.slice(ens, [0, 1], [-1, -1])
# #     print("tf.shape(ens):" + str(tf.shape(ens)))
#     return ens
  
#   def multi_infer(self, mems_before_last, last_token, steps):
#     ''' here mems_before_last shape must be [n_layer memory_length 1 feature_size] '''
#     output, _, _, _, _ = self.transformer_model.transformer(last_token, tf.zeros_like(last_token)-1, mems_before_last, tf.ones_like(last_token), is_training=0)
# #     all_mems = update_recent_fixed_length_memory(mems_before_last, old_new_mems)
#     ''' output shape should be [predict_length batch_size feature_size] '''
#     ''' output shape should be [1 1 feature_size] '''
#     
#     def multi_infer_cond(i, i_len, *_):
#       return tf.less(i, i_len)
#     
#     def multi_infer_body(i, i_len, o_log_probs, o_ens):
#       transfer_i = tf.expand_dims(tf.expand_dims(i, 0), 1)
#       t_h = self.multi_decode_model.multi_position_transfer.transfer(transfer_i, output)
#       ''' t_h shape: [1, 1, feature_size] '''
#       o_log_probs_of_this_node, o_ens_of_this_node = self.multi_decode_model.loss_calculator.only_compute_predictions(t_h)
#       o_log_probs_of_this_node = tf.squeeze(o_log_probs_of_this_node)
#       o_ens_of_this_node = tf.squeeze(o_ens_of_this_node)
#       o_log_probs = tf.concat([o_log_probs, [o_log_probs_of_this_node]], axis=0)
#       o_ens = tf.concat([o_ens, [o_ens_of_this_node]], axis=0)
#       return (i+1, i_len, o_log_probs, o_ens)
#     
#     i = tf.constant(0, int_type)
#     o_log_probs = tf.zeros([0, top_ks[-1]], float_type)
#     o_ens = tf.zeros([0, top_ks[-1]], int_type)
#     r_steps = tf.cast(tf.minimum(multi_infer_num, steps), int_type)
#     _, _, o_log_probs, o_ens = tf.while_loop(multi_infer_cond, multi_infer_body, [i, r_steps, o_log_probs, o_ens], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, top_ks[-1]]), tf.TensorShape([None, top_ks[-1]])], parallel_iterations=1)
#     _, computed_en_seqs = dp_compute_en_seqs_from_distinct_parallel_tokens(o_log_probs, o_ens)
#     ''' ensure computed_en_seqs to shape: [top_ks[-1], steps] '''
# #     print("str(tf.shape(computed_en_seqs)):" + str(tf.shape(computed_en_seqs)))
#     
#     left_steps = steps - r_steps
#     assert left_steps <= 0
# #     if left_steps.numpy() > 0:
# #       cat_last_token = tf.tile(last_token, [1, top_ks[-1]])
# #       temp_all_tokens = tf.concat([cat_last_token, tf.transpose(computed_en_seqs, perm=[1, 0])], axis=0)
# #       before_last_temp_all_tokens = tf.slice(temp_all_tokens, [0, 0], [tf.shape(temp_all_tokens)[0]-1, -1])
# #       last_temp_all_tokens = temp_all_tokens[-1:, :]
# #       new_mems_before_last = []
# #       for j in range(n_layer):
# #         new_mems_before_last.append(tf.tile(mems_before_last[j], [1, top_ks[-1], 1]))
# #       _, _, _, _, new_mems = self.transformer_model.transformer(before_last_temp_all_tokens, tf.zeros_like(before_last_temp_all_tokens)-1, new_mems_before_last, tf.ones_like(before_last_temp_all_tokens), is_training=0, mem_len=oracle_mem_len)
# #       new_temp_all_mems = update_recent_fixed_length_memory(new_mems_before_last, new_mems)
# #       infer_ens = self.infer(new_temp_all_mems, last_temp_all_tokens, left_steps, initial_batch_size_is_one=False, initial_probs=computed_probs, initial_ens=computed_en_seqs)
# #       computed_en_seqs = infer_ens# tf.concat([computed_en_seqs, infer_ens], axis=1)
#     return computed_en_seqs
  
  
  







  
  












