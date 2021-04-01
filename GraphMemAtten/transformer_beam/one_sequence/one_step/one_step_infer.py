import tensorflow as tf
import numpy as np
from meta_info.hyper_parameter import oracle_mem_len,\
  additional_filter_memory_when_beam_step_inferring,\
  memory_train_test_beam_consistent, all_skt_h_num, all_skt_par_hint_to_id,\
  all_skt_id_to_str, multi_infer_num, n_skt
from utils.memory_util import update_recent_fixed_length_memory,\
  get_specified_varied_length_memory
from meta_info.non_hyper_constant import unk_id, parent_info_length, skt_dft,\
  np_int_type, int_type


class OneStepStandInfer():
  
  def __init__(self, transformer_model, mems_before_last, last_token):
    self.transformer_model = transformer_model
    self.mems = list(mems_before_last)
    self.l_token = last_token
#     print("tf.shape(self.l_token):" + str(tf.shape(self.l_token)))
    self.i = -1
  
  def infer_one_step(self, par_hint, pos_hint):
    self.i += 1
    output, _, _, _, new_mems = self.transformer_model.transformer(self.l_token, tf.zeros_like(self.l_token)-1, self.mems, tf.ones_like(self.l_token), par_hint, pos_hint, is_training=False, calculate_loss=False)
    
    o_ens_probs, o_ens_of_this_node = self.get_loss_caculator().only_compute_predictions(output, par_hint, pos_hint)
    
    new_mems = update_recent_fixed_length_memory(self.mems, new_mems)
    if additional_filter_memory_when_beam_step_inferring:
      new_mems = get_specified_varied_length_memory(new_mems, -1, oracle_mem_len, memory_train_test_beam_consistent)
    self.mems = new_mems
    return o_ens_probs, o_ens_of_this_node
  
  def record_just_inferred_en(self, inferred_en):
    self.l_token = inferred_en
    
  def get_loss_caculator(self):
    return self.transformer_model.loss_calculator
    

class OneStepMultiInfer():
  
  def __init__(self, transformer_model, multi_decode_model, mems_before_last, last_token):
    self.transformer_model = transformer_model
    self.multi_decode_model = multi_decode_model
    self.mems_before_last = mems_before_last
    self.last_token = last_token
    self.i = -1
  
  def infer_one_step(self, par_hint, pos_hint):
    self.i += 1
    if self.i == 0:
      output, _, _, _, _ = self.transformer_model.transformer(self.last_token, tf.zeros_like(self.last_token)-1, self.mems_before_last, tf.ones_like(self.last_token), par_hint, pos_hint, is_training=0, calculate_loss=False)
      self.output = output
    
    transfer_i = tf.expand_dims(tf.expand_dims(self.i, 0), 1)
    t_h = self.multi_decode_model.multi_position_transfer.transfer(transfer_i, self.output)
    o_ens_probs, o_ens_of_this_node = self.get_loss_caculator().only_compute_predictions(t_h, par_hint, pos_hint)
    return o_ens_probs, o_ens_of_this_node
  
  def record_just_inferred_en(self, inferred_en):
    pass
  
  def get_loss_caculator(self):
    return self.multi_decode_model.loss_calculator


def framework_skt_infer(inferrer, parent_hint, position_hint, steps): # @UnusedVariable
  ''' here mems_before_last shape must be [n_layer memory_length 1 feature_size] '''
#     all_mems = update_recent_fixed_length_memory(mems_before_last, old_new_mems)
  ''' output shape should be [predict_length batch_size feature_size] '''
  ''' output shape should be [1 1 feature_size] '''
  
#     def multi_infer_cond(i, i_len, *_):
#       return tf.less(i, i_len)
  
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
  one_computed_en_seq = []
  computed_en_seqs = [one_computed_en_seq]
  
  r_steps = tf.cast(tf.minimum(multi_infer_num, steps), int_type)
  np_r_steps = r_steps.numpy()
  guide = np.zeros([np_r_steps], np_int_type)
  
  en_stack = []
  h_index_stack = []
  h_num_stack = []
  
#     for _ in range(parent_info_length):
#       en_stack.append(0)
#       h_index_stack.append(0)
#       h_num_stack.append(0)
#     h_index_stack[-1] -= 1
  
  i = 0
  while (i < np_r_steps):
#     t_h = inferrer.infer_one_step()
    ''' t_h shape: [1, 1, feature_size] '''
    
    par_hint_str = ""
    compensate_size = parent_info_length - len(en_stack)
    if (compensate_size > 0):
      t = 0
      while (t < compensate_size):
        par_hint_str += (skt_dft + ":0#")
        t += 1
    ot = max(0, len(en_stack) - parent_info_length)
    while (ot < len(en_stack)):
      en = en_stack[ot]
      en_str = all_skt_id_to_str[en]
      par_hint_str += en_str + ":" + str(h_index_stack[ot]) + "#"
      ot += 1
    
    if par_hint_str in all_skt_par_hint_to_id:
      par_hint_id = all_skt_par_hint_to_id[par_hint_str]
    else:
      par_hint_id = unk_id
    
#     if (parent_hint[i] != par_hint_id):
#       print("i:" + str(i) + "#parent_hint[i]:" + str(parent_hint[i].numpy()) + "#all_skt_par_hint_id[parent_hint[i]]:" + all_skt_par_hint_id[parent_hint[i].numpy()] + "#par_hint_id:" + str(par_hint_id) + "#all_skt_par_hint_id[par_hint_id]:" + all_skt_par_hint_id[par_hint_id])
    
    par_hint = [[par_hint_id]]
    pos_hint = [[i * 2 + 1]]
#     print("par_hint:" + str(par_hint))
#     p_op = tf.print("tf.shape(t_h):", tf.shape(t_h), "tf.shape(par_hint):", tf.shape(par_hint))
#     with tf.control_dependencies([p_op]):
#     _, o_ens_of_this_node = inferrer.get_loss_caculator().only_compute_predictions(t_h, par_hint, pos_hint)
    _, o_ens_of_this_node = inferrer.infer_one_step(par_hint, pos_hint)
    guide_en_tf = o_ens_of_this_node[0][0][guide[i]]
#     p_op = tf.print("o_ens_of_this_node:", o_ens_of_this_node, "par_hint:", par_hint)
#     with tf.control_dependencies([p_op]):
    guide_en = guide_en_tf.numpy()
    one_computed_en_seq.append(guide_en)
    inferrer.record_just_inferred_en([[guide_en]])
    
    ''' handle back-trace, may back a few steps '''
    en_stack.append(guide_en)
    h_index_stack.append(0)
#     print("guide_en:" + str(guide_en) + "#n_skt:" + str(n_skt))
    if guide_en < n_skt:
      h_num = all_skt_h_num[guide_en]
    else:
      h_num = 0
    h_num_stack.append(h_num)
    
    if (h_num == 0):
      ''' begin back trace and delete '''
      s_last = len(h_num_stack) - 1
      while (s_last >= 0):
        if (h_num_stack[s_last] >= h_index_stack[s_last]):# or h_num_stack[s_last] == 0
          h_num_stack.pop()
          h_index_stack.pop()
          en_stack.pop()
          s_last-=1
        else:
          break;
      assert len(h_index_stack) - 1 == s_last
#       print("len(h_index_stack):" + str(len(h_index_stack)))
      if len(h_index_stack) > 0:
        h_index_stack[-1] += 1
    
    i += 1
  
#     i = tf.constant(0, int_type)
#     o_log_probs = tf.zeros([0, top_ks[-1]], float_type)
#     o_ens = tf.zeros([0, top_ks[-1]], int_type)
#     _, _, o_log_probs, o_ens = tf.while_loop(multi_infer_cond, multi_infer_body, [i, r_steps, o_log_probs, o_ens], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, top_ks[-1]]), tf.TensorShape([None, top_ks[-1]])], parallel_iterations=1)
#     _, computed_en_seqs = dp_compute_en_seqs_from_distinct_parallel_tokens(o_log_probs, o_ens)
#     ''' ensure computed_en_seqs to shape: [top_ks[-1], steps] '''
  
  left_steps = steps - r_steps
  assert left_steps <= 0
  return computed_en_seqs


def framework_token_infer(inferrer, parent_hint, position_hint, steps):
  ''' here mems_before_last shape must be [n_layer memory_length 1 feature_size] '''
#     all_mems = update_recent_fixed_length_memory(mems_before_last, old_new_mems)
  ''' output shape should be [predict_length batch_size feature_size] '''
  ''' output shape should be [1 1 feature_size] '''
  
#     def multi_infer_cond(i, i_len, *_):
#       return tf.less(i, i_len)
  
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
  one_computed_en_seq = []
  computed_en_seqs = [one_computed_en_seq]
  
  r_steps = tf.cast(tf.minimum(multi_infer_num, steps), int_type)
  np_r_steps = r_steps.numpy()
  guide = np.zeros([np_r_steps], np_int_type)
  
#     for _ in range(parent_info_length):
#       en_stack.append(0)
#       h_index_stack.append(0)
#       h_num_stack.append(0)
#     h_index_stack[-1] -= 1
  
  i = 0
  while (i < np_r_steps):
    par_hint_id = parent_hint[i]
    pos_hint_id = position_hint[i]
    assert(pos_hint_id.numpy() == i * 2 + 2)
    
#     t_h = inferrer.infer_one_step()
    ''' t_h shape: [1, 1, feature_size] '''
    
    par_hint = [[par_hint_id]]
    pos_hint = [[pos_hint_id]]
#     print("par_hint:" + str(par_hint))
#     p_op = tf.print("tf.shape(t_h):", tf.shape(t_h), "tf.shape(par_hint):", tf.shape(par_hint))
#     with tf.control_dependencies([p_op]):
#     _, o_ens_of_this_node = inferrer.get_loss_caculator().only_compute_predictions(t_h, par_hint, pos_hint)
    _, o_ens_of_this_node = inferrer.infer_one_step(par_hint, pos_hint)
    guide_en_tf = o_ens_of_this_node[0][0][guide[i]]
#     p_op = tf.print("o_ens_of_this_node:", o_ens_of_this_node, "par_hint:", par_hint)
#     with tf.control_dependencies([p_op]):
    guide_en = guide_en_tf.numpy()
    one_computed_en_seq.append(guide_en)
    inferrer.record_just_inferred_en([[guide_en]])
    
    i += 1
  
#     i = tf.constant(0, int_type)
#     o_log_probs = tf.zeros([0, top_ks[-1]], float_type)
#     o_ens = tf.zeros([0, top_ks[-1]], int_type)
#     _, _, o_log_probs, o_ens = tf.while_loop(multi_infer_cond, multi_infer_body, [i, r_steps, o_log_probs, o_ens], [tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape([None, top_ks[-1]]), tf.TensorShape([None, top_ks[-1]])], parallel_iterations=1)
#     _, computed_en_seqs = dp_compute_en_seqs_from_distinct_parallel_tokens(o_log_probs, o_ens)
#     ''' ensure computed_en_seqs to shape: [top_ks[-1], steps] '''
  
  left_steps = steps - r_steps
  assert left_steps <= 0
  return computed_en_seqs









