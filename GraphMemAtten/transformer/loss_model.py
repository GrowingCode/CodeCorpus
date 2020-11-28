import tensorflow as tf
from meta_info.non_hyper_constant import float_type, top_ks
from meta_info.hyper_parameter import d_model, n_token, d_embed, multi_infer_num
from utils.initialize_util import random_normal_variable_initializer,\
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
  def mask_adaptive_logsoftmax(self, hidden, target, valid_mask, compute_prediction):
  
    output = generate_logit(hidden, self.token_output_w, self.token_output_softmax_b, self.proj_w)
    
#     nll = None
#     if not compute_prediction:
    nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                         logits=output)
    
    ''' ['tf.shape(output):', [128 6 27]] '''
    ''' ['tf.shape(nll):', [128 6]] '''
    nll = nll * tf.cast(valid_mask, float_type)
    
#     if return_mean:
#       nll = tf.reduce_mean(input_tensor=nll)
    nll = tf.reduce_sum(input_tensor=nll)
    
#       print("n_token:" + str(n_token))
#       target_max = tf.reduce_max(target)
#       print("target_max:" + str(target_max))
#       all_nll_nans = tf.reduce_sum(tf.cast(tf.math.is_nan(nll), int_type))
#       print("all_nll_nans:" + str(all_nll_nans))
#     print("reduced nll:" + str(nll))
      
    probs, predictions = None, None
    if compute_prediction:
      ''' ['tf.shape(predictions):', [128 6 top_ks[-1]]] '''
      t_probs = tf.math.log(tf.nn.softmax(output, axis=2))
      probs, predictions = tf.nn.top_k(t_probs, top_ks[-1])
#     print("nll:" + str(nll))
    return probs, predictions, nll
  
  def only_compute_predictions(self, t_h):
    ''' t_h shape: [tgt_size, batch_size, feature_size] actually [1, 1, feature_size] '''
    ''' output shape: [tgt_size, batch_size, top_ks[-1]] '''
    output = generate_logit(t_h, self.token_output_w, self.token_output_softmax_b, self.proj_w)
    t_probs = tf.math.log(tf.nn.softmax(output, axis=2))
    probs, predictions = tf.nn.top_k(t_probs, top_ks[-1])
    return probs, predictions


def generate_logit(x, W, b, proj):
      y = x
      if proj is not None:
        y = tf.einsum('ibd,ed->ibe', y, proj)
      return tf.einsum('ibd,nd->ibn', y, W) + b


class MultiLossCalculator(tf.keras.Model):
  
  def __init__(self):
    super(MultiLossCalculator, self).__init__()
    self.proj_w = None
    if d_model != d_embed:
      self.proj_w = tf.Variable(random_normal_variable_initializer([d_embed, d_model]))
    self.token_output_w = tf.Variable(random_normal_variable_initializer([multi_infer_num+1, n_token, d_model]))
    self.token_output_softmax_b = tf.Variable(zero_variable_initializer([multi_infer_num+1, n_token]))
  
  # return_mean=True
  def mask_adaptive_logsoftmax(self, hidden, target, relative_to_part_first, valid_mask, compute_prediction):
    
    imd = tf.einsum("ibe,pne->ibpn", hidden, self.token_output_w)
    imd_p = tf.one_hot(relative_to_part_first, multi_infer_num+1)
    output_base = tf.einsum("ibpn,ibp->ibn", imd, imd_p)
    output_bias = tf.einsum("ibp,pn->ibn", imd_p, self.token_output_softmax_b)
    
    output = output_base + output_bias
    
#     output = generate_logit(hidden, self.token_output_w, self.token_output_softmax_b, self.proj_w)
    
#     nll = None
#     if not compute_prediction:
    nll = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,
                                                         logits=output)
    
    ''' ['tf.shape(output):', [128 6 27]] '''
    ''' ['tf.shape(nll):', [128 6]] '''
    nll = nll * tf.cast(valid_mask, float_type)
    
#     if return_mean:
#       nll = tf.reduce_mean(input_tensor=nll)
    nll = tf.reduce_sum(input_tensor=nll)
    
#       print("n_token:" + str(n_token))
#       target_max = tf.reduce_max(target)
#       print("target_max:" + str(target_max))
#       all_nll_nans = tf.reduce_sum(tf.cast(tf.math.is_nan(nll), int_type))
#       print("all_nll_nans:" + str(all_nll_nans))
#     print("reduced nll:" + str(nll))
      
    probs, predictions = None, None
    if compute_prediction:
      ''' ['tf.shape(predictions):', [128 6 top_ks[-1]]] '''
      t_probs = tf.math.log(tf.nn.softmax(output, axis=2))
      probs, predictions = tf.nn.top_k(t_probs, top_ks[-1])
#     print("nll:" + str(nll))
    return probs, predictions, nll







