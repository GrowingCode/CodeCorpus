import tensorflow as tf
from utils.batch_gather_util import batch_gather


class MultiDecodeModel(tf.keras.Model):
  
  def __init__(self, transformer_model, multi_position_transfer):
    super(MultiDecodeModel, self).__init__()
    self.transformer_model = transformer_model
    self.multi_position_transfer = multi_position_transfer
  
  def multi_decode(self, dec_inp, target, relative_to_part_first, all_outputs, mems, is_training):
    target_length = tf.shape(target)[0]
    '''
    dec_inp shape:[target_length, batch_size]
    target shape: [target_length, batch_size]
    relative_to_part_first shape: [target_length, batch_size]
    '''
    ''' all_outputs shape: [all_already_outputs_length(may drop the initial) batch_size feature_size] '''
    outputs, _, _, _, new_mems = self.transformer_model.transformer(dec_inp, target, mems, is_training=is_training)
    ''' outputs shape: [target_length batch_size feature_size] '''
    new_all_outputs = tf.concat([all_outputs, outputs], axis=0)
    
    outs_positions = tf.range(target_length) - target_length - relative_to_part_first - 1
    used_outputs = batch_gather(new_all_outputs, outs_positions)
#     used_outputs = tf.gather_nd(params=new_all_outputs, indices=outs_positions)
    
    ''' used_outputs shape: [target_length batch_size feature_size] '''
    transferred_outputs = self.transformer_model.transfer(used_outputs, outs_positions)
    probs, predictions, loss = self.transformer_model.mask_adaptive_logsoftmax(transferred_outputs, target)
    
    return transferred_outputs, probs, predictions, loss, new_mems





