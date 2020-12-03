import tensorflow as tf
from meta_info.hyper_parameter import multi_infer_num, d_embed,\
  multi_position_transfer_layer, use_simple_multi_infer_mode
from utils.initialize_util import random_normal_variable_initializer


class MultiPositionTransfer(tf.keras.Model):
  
  def __init__(self):
    super(MultiPositionTransfer, self).__init__()
    self.t_layers = []
    for _ in range(multi_position_transfer_layer):
      self.t_layers.append(MultiPositionTransferLayer())
  
  def transfer(self, positions, outputs):
    r_outputs = outputs
    for i in range(multi_position_transfer_layer):
      r_outputs = self.t_layers[i].transfer(positions, r_outputs)
    return r_outputs
  
  
class MultiPositionTransferLayer(tf.keras.Model):
  
  def __init__(self):
    super(MultiPositionTransferLayer, self).__init__()
    if use_simple_multi_infer_mode:
      self.out_fs = LinearTransferFeatures()
    else:
      self.self_forget_fs = LinearTransferFeatures()
      self.logit_fs = LinearTransferFeatures()
      self.logit_forget_fs = LinearTransferFeatures()
      self.out_forget_fs = LinearTransferFeatures()
    
  def transfer(self, positions, outputs):
    if use_simple_multi_infer_mode:
      new_res = self.out_fs.linear(positions, outputs)
    else:
      self_forget = self.self_forget_fs.linear(positions, outputs)
      logit = self.logit_fs.linear(positions, outputs)
      logit_forget = self.logit_forget_fs.linear(positions, outputs)
      out_forget = self.out_forget_fs.linear(positions, outputs)
      new_res = (outputs * tf.nn.sigmoid(self_forget) + 
               tf.nn.tanh(logit) * tf.nn.sigmoid(logit_forget))
      new_res = tf.nn.tanh(new_res) * tf.nn.sigmoid(out_forget)
    return new_res
    
    
class LinearTransferFeatures(tf.keras.Model):
  
  def __init__(self):
    super(LinearTransferFeatures, self).__init__()
    self.multi_transfer_w = tf.Variable(random_normal_variable_initializer([multi_infer_num + 1, d_embed, d_embed]))
    self.multi_transfer_b = tf.Variable(random_normal_variable_initializer([multi_infer_num + 1, d_embed]))
    
  def linear(self, positions, outputs):
    ''' outputs shape: [target_length, batch_size, feature_size] '''
    ''' positions shape: [target_length, batch_size] '''
    ''' positions need to transfer to positions embedding '''
    ''' positions embedding shape: [target_length, batch_size, feature_size, feature_size] '''
    r_positions = tf.where(positions < multi_infer_num, positions, tf.zeros_like(positions) + multi_infer_num)
    positions_w_embedding = tf.nn.embedding_lookup(self.multi_transfer_w, r_positions)
    transferred_outputs = tf.einsum('ibde,ibd->ibe', positions_w_embedding, outputs)
    positions_b_embedding = tf.nn.embedding_lookup(self.multi_transfer_b, r_positions)
    transferred_outputs = transferred_outputs + positions_b_embedding
    ''' transferred_outputs shape: [target_length, batch_size, feature_size] '''
    return transferred_outputs
    
    
    
  
  
  
  
  
  
  
  




