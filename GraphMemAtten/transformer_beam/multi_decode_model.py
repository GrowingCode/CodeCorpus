import tensorflow as tf
from utils.batch_gather_util import batch_gather
from transformer.loss_model import LossCalculator


class MultiDecodeModel(tf.keras.Model):
  
  def __init__(self, transformer_model, multi_position_transfer):
    super(MultiDecodeModel, self).__init__()
    self.transformer_model = transformer_model
    self.multi_position_transfer = multi_position_transfer
#     self.loss_calculator = transformer_model.loss_calculator
    self.loss_calculator = LossCalculator()
  
  def multi_decode(self, dec_inp, target, relative_to_part_first, all_outputs, mems, valid_mask, is_training):
#     relative_to_part_first = tf.zeros_like(relative_to_part_first)
    
    target_length = tf.shape(target)[0]
    '''
    dec_inp shape:[target_length, batch_size]
    target shape: [target_length, batch_size]
    relative_to_part_first shape: [target_length, batch_size]
    '''
    ''' all_outputs shape: [all_already_outputs_length(may drop the initial) batch_size feature_size] '''
    outputs, _, _, _, new_mems = self.transformer_model.transformer(dec_inp, target, mems, valid_mask, is_training=is_training)
    ''' outputs shape: [target_length batch_size feature_size] '''
    new_all_outputs = tf.concat([all_outputs, outputs], axis=0)
    
    all_o_length = tf.shape(new_all_outputs)[0]
    outs_positions = tf.tile(tf.expand_dims(tf.range(target_length), axis=1), [1, 6]) - target_length - relative_to_part_first + all_o_length
    used_outputs = batch_gather(new_all_outputs, outs_positions)
#     used_outputs = tf.gather_nd(params=new_all_outputs, indices=outs_positions)
    
    ''' used_outputs shape: [target_length batch_size feature_size] '''
#     transferred_outputs = used_outputs
    transferred_outputs = self.multi_position_transfer.transfer(relative_to_part_first, used_outputs)
    probs, predictions, loss = self.loss_calculator.mask_adaptive_logsoftmax(transferred_outputs, target, valid_mask, is_training == False)
    
    return new_all_outputs, probs, predictions, loss, new_mems





