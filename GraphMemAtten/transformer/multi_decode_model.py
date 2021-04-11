import tensorflow as tf
from transformer.loss_model import LossCalculator
from utils.batch_gather_util import batch_gather
from meta_info.hyper_parameter import multi_infer_train_to_predict_unk


class MultiDecodeModel(tf.keras.Model):
  
  def __init__(self, transformer_model, multi_position_transfer):
    super(MultiDecodeModel, self).__init__()
    self.transformer_model = transformer_model
    self.multi_position_transfer = multi_position_transfer
#     self.loss_calculator = transformer_model.loss_calculator
    self.loss_calculator = LossCalculator()
#     self.multi_loss_calculator = MultiLossCalculator()
  
  def multi_decode(self, dec_inp, target, relative_to_part_first, all_outputs, mems, valid_mask, parent_hint, position_hint, is_training):
#     relative_to_part_first = tf.zeros_like(relative_to_part_first)
    
    target_length = tf.shape(target)[0]
    b_size = tf.shape(target)[1]
#     assert b_size.numpy() <= batch_size
    '''
    dec_inp shape:[target_length, batch_size]
    target shape: [target_length, batch_size]
    relative_to_part_first shape: [target_length, batch_size]
    '''
    ''' all_outputs shape: [all_already_outputs_length(may drop the initial) batch_size feature_size] '''
    outputs, _, _, _, new_mems = self.transformer_model.transformer(dec_inp, target, mems, valid_mask, parent_hint, position_hint, is_training=is_training, calculate_loss=False)
#     if stop_gradient_for_transformer_output_in_multi_decode:
#       outputs = tf.stop_gradient(outputs)
    ''' outputs shape: [target_length batch_size feature_size] '''
    new_all_outputs = tf.concat([all_outputs, outputs], axis=0)
    
    all_o_length = tf.shape(new_all_outputs)[0]
    outs_positions = tf.tile(tf.expand_dims(tf.range(target_length), axis=1), [1, b_size]) - target_length - relative_to_part_first + all_o_length
    used_outputs = batch_gather(new_all_outputs, outs_positions)
#     used_outputs = tf.gather_nd(params=new_all_outputs, indices=outs_positions)
    
    ''' used_outputs shape: [target_length batch_size feature_size] '''
#     transferred_outputs = used_outputs
    transferred_outputs = self.multi_position_transfer.transfer(relative_to_part_first, used_outputs)
    probs, predictions, loss = self.loss_calculator.mask_adaptive_logsoftmax(transferred_outputs, target, valid_mask, parent_hint, position_hint, is_training == False, multi_infer_train_to_predict_unk)
#     probs, predictions, loss = self.multi_loss_calculator.mask_adaptive_logsoftmax(transferred_outputs, target, relative_to_part_first, valid_mask, is_training == False)
    
    return new_all_outputs, probs, predictions, loss, new_mems





