import tensorflow as tf
from utils.initialize_util import random_uniform_variable_initializer
from meta_info.hyper_parameter import multi_infer_num, d_embed


class MultiPositionTransfer(tf.keras.Model):
  
  def __init__(self):
    super(MultiPositionTransfer, self).__init__()
    self.multi_transfer_parameters = tf.Variable(random_uniform_variable_initializer(258, 578, [multi_infer_num, d_embed, d_embed]))
    
  def transfer(self, positions, outputs):
    ''' positions shape: [target_length, batch_size] '''
    ''' positions need to transfer to positions embedding '''
    ''' positions embedding shape: [target_length, batch_size, feature_size, feature_size] '''
    positions_embedding = tf.nn.embedding_lookup(self.multi_transfer_parameters, positions)
    
    ''' outputs shape: [target_length, batch_size, feature_size] '''
    transferred_outputs = tf.einsum('ibde,ibd->ibe', positions_embedding, outputs)
    return transferred_outputs
  
  




