import tensorflow as tf


class MultiPositionTransfer():
  
  def __init__(self):
    
    pass
  
  def transfer(self, positions, outputs):
    ''' positions shape: [target_length, batch_size] '''
    ''' positions need to transfer to positions embedding '''
    ''' positions embedding shape: [target_length, batch_size, feature_size] '''
    tf.nn.embedding_lookup(lookup_table, x)
    
    ''' outputs shape: [target_length, batch_size, feature_size] '''
    transfer_mat = self.multi_transfer_parameters[i]
    t_h = tf.matmul(h, transfer_mat)
    
    ''' outputs and positions embedding should be each matmul '''
    
    pass
  
  




