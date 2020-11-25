import tensorflow as tf
from meta_info.hyper_parameter import multi_infer_num, d_embed
from utils.initialize_util import random_normal_variable_initializer


class MultiPositionTransfer(tf.keras.Model):
  
  def __init__(self):
    super(MultiPositionTransfer, self).__init__()
    self.multi_transfer_parameters = tf.Variable(random_normal_variable_initializer([multi_infer_num + 1, d_embed, d_embed]))
    
  def transfer(self, positions, outputs):
    ''' outputs shape: [target_length, batch_size, feature_size] '''
    ''' positions shape: [target_length, batch_size] '''
    ''' positions need to transfer to positions embedding '''
    ''' positions embedding shape: [target_length, batch_size, feature_size, feature_size] '''
    r_positions = tf.where(positions < multi_infer_num, positions, tf.zeros_like(positions) + multi_infer_num)
    positions_embedding = tf.nn.embedding_lookup(self.multi_transfer_parameters, r_positions)
    transferred_outputs = tf.einsum('ibde,ibd->ibe', positions_embedding, outputs)
    ''' transferred_outputs shape: [target_length, batch_size, feature_size] '''
    return transferred_outputs
  
  




