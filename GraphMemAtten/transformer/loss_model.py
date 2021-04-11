from meta_info.hyper_parameter import d_model, n_token, d_embed, \
  all_skt_hint_mask, all_skt_position_hint_mask, consider_position_hint
from meta_info.non_hyper_constant import top_ks
import tensorflow as tf
from utils.initialize_util import random_normal_variable_initializer, \
  zero_variable_initializer


class LossCalculator(tf.keras.Model):
  
  def __init__(self):
    super(LossCalculator, self).__init__()
    self.proj_w = None
    if d_model != d_embed:
      self.proj_w = tf.Variable(random_normal_variable_initializer([d_embed, d_model]))
    self.token_output_w = tf.Variable(random_normal_variable_initializer([n_token, d_model]))
    self.token_output_softmax_b = tf.Variable(zero_variable_initializer([n_token]))
  
  # return_mean=True
  def mask_adaptive_logsoftmax(self, hidden, target, valid_mask, parent_hint, position_hint, compute_prediction, train_to_predict_unk=False, calculate_loss=True):
    # prediction_mask, 
    r_output = self.logit_with_hint(hidden, parent_hint, position_hint)
    
#     if debug_assert:
#       o_hot = tf.nn.embedding_lookup(n_token_one_hot, target)
#       one_sum = tf.einsum('ibd,ibd->ib', o_hot, prediction_mask)
#       result = tf.reduce_all(tf.equal(one_sum, tf.constant(1, int_type)))
#       assert result.numpy() == True
      
#     nll = None
#     if not compute_prediction:
#     p_op = tf.print("tf.shape(hidden):", tf.shape(hidden), "tf.shape(valid_mask):", tf.shape(valid_mask), "tf.shape(parent_hint):", tf.shape(parent_hint), "tf.shape(target):", tf.shape(target), "tf.shape(r_output):", tf.shape(r_output))
#     with tf.control_dependencies([p_op]):
    if calculate_loss:
      nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                           logits=r_output)
      ''' ['tf.shape(output):', [128 6 27]] '''
      ''' ['tf.shape(nll):', [128 6]] '''
  #     print("valid_mask:" + str(valid_mask))
      if train_to_predict_unk:
        r_nll = nll
      else:
        r_nll = tf.where(tf.equal(valid_mask, 1), nll, tf.zeros_like(nll))
  #     r_nll = nll * tf.cast(valid_mask, float_type)
  #     if return_mean:
  #       nll = tf.reduce_mean(input_tensor=nll)
      r_nll_sum = tf.reduce_sum(input_tensor=r_nll)
    else:
      r_nll_sum = 0.0
#       print("n_token:" + str(n_token))
#       target_max = tf.reduce_max(target)
#       print("target_max:" + str(target_max))
#       all_nll_nans = tf.reduce_sum(tf.cast(tf.math.is_nan(nll), int_type))
#       print("all_nll_nans:" + str(all_nll_nans))
#     print("reduced nll:" + str(nll))
      
    probs, predictions = None, None
    if compute_prediction:
      ''' ['tf.shape(predictions):', [128 6 top_ks[-1]]] '''
      t_probs = tf.math.log(tf.nn.softmax(r_output, axis=2))
      probs, predictions = tf.nn.top_k(t_probs, top_ks[-1])
#     print("r_nll:" + str(r_nll))
    return probs, predictions, r_nll_sum
  
  def only_compute_predictions(self, t_h, parent_hint, position_hint):
    ''' t_h shape: [tgt_size, batch_size, feature_size] actually [1, 1, feature_size] '''
    ''' predictions shape: [tgt_size, batch_size, top_ks[-1]] '''
    
    r_output = self.logit_with_hint(t_h, parent_hint, position_hint)
    
#     p_op = tf.print("tf.shape(t_h):", tf.shape(t_h), "tf.shape(r_output):", tf.shape(r_output))
#     with tf.control_dependencies([p_op]):
    t_probs = tf.math.log(tf.nn.softmax(r_output, axis=2))
    
    probs, predictions = tf.nn.top_k(t_probs, top_ks[-1])
    return probs, predictions
  
  def logit_with_hint(self, hidden, parent_hint, position_hint):
#     prediction_mask = tf.gather(hint_mask, parent_hint)
    output = generate_logit(hidden, self.token_output_w, self.token_output_softmax_b, self.proj_w)
    if consider_position_hint > 0:
      parent_prediction_mask = tf.nn.embedding_lookup(all_skt_hint_mask, parent_hint)
      ''' ['tf.shape(prediction_mask):', [128 6 27]] '''
      position_prediction_mask = tf.nn.embedding_lookup(all_skt_position_hint_mask, position_hint)
      if consider_position_hint == 1:
        prediction_mask = parent_prediction_mask
      if consider_position_hint == 2:
        prediction_mask = tf.bitwise.bitwise_and(parent_prediction_mask, position_prediction_mask)
      elif consider_position_hint == 3:
        prediction_mask = position_prediction_mask
      else:
        assert False
      r_output = tf.where(tf.equal(prediction_mask, 1), output, -1e30*tf.ones_like(output))
    else:
      r_output = output
#     p_op = tf.print("tf.shape(parent_hint):", tf.shape(parent_hint), "tf.shape(hidden):", tf.shape(hidden), "tf.shape(prediction_mask):", tf.shape(prediction_mask), "tf.shape(output):", tf.shape(output))
#     with tf.control_dependencies([p_op]):
    return r_output
  

def generate_logit(x, W, b, proj):
      y = x
      if proj is not None:
        y = tf.einsum('ibd,ed->ibe', y, proj)
      return tf.einsum('ibd,nd->ibn', y, W) + b


# class MultiLossCalculator(tf.keras.Model):
#   
#   def __init__(self):
#     super(MultiLossCalculator, self).__init__()
#     self.proj_w = None
#     if d_model != d_embed:
#       self.proj_w = tf.Variable(random_normal_variable_initializer([d_embed, d_model]))
#     self.token_output_w = tf.Variable(random_normal_variable_initializer([multi_infer_num+1, n_token, d_model]))
#     self.token_output_softmax_b = tf.Variable(zero_variable_initializer([multi_infer_num+1, n_token]))
#   
#   # return_mean=True
#   def mask_adaptive_logsoftmax(self, hidden, target, relative_to_part_first, valid_mask, compute_prediction):
#     
#     imd = tf.einsum("ibe,pne->ibpn", hidden, self.token_output_w)
#     imd_p = tf.one_hot(relative_to_part_first, multi_infer_num+1)
#     output_base = tf.einsum("ibpn,ibp->ibn", imd, imd_p)
#     output_bias = tf.einsum("ibp,pn->ibn", imd_p, self.token_output_softmax_b)
#     
#     output = output_base + output_bias
#     
# #     output = generate_logit(hidden, self.token_output_w, self.token_output_softmax_b, self.proj_w)
#     
# #     nll = None
# #     if not compute_prediction:
#     nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
#                                                          logits=output)
#     
#     ''' ['tf.shape(output):', [128 6 27]] '''
#     ''' ['tf.shape(nll):', [128 6]] '''
#     nll = nll * tf.cast(valid_mask, float_type)
#     
# #     if return_mean:
# #       nll = tf.reduce_mean(input_tensor=nll)
#     nll = tf.reduce_sum(input_tensor=nll)
#     
# #       print("n_token:" + str(n_token))
# #       target_max = tf.reduce_max(target)
# #       print("target_max:" + str(target_max))
# #       all_nll_nans = tf.reduce_sum(tf.cast(tf.math.is_nan(nll), int_type))
# #       print("all_nll_nans:" + str(all_nll_nans))
# #     print("reduced nll:" + str(nll))
#       
#     probs, predictions = None, None
#     if compute_prediction:
#       ''' ['tf.shape(predictions):', [128 6 top_ks[-1]]] '''
#       t_probs = tf.math.log(tf.nn.softmax(output, axis=2))
#       probs, predictions = tf.nn.top_k(t_probs, top_ks[-1])
# #     print("nll:" + str(nll))
#     return probs, predictions, nll







