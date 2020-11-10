from meta_info.hyper_parameter import oracle_mem_len, n_layer, d_model, initial_memory_trainable
from meta_info.non_hyper_constant import float_type, multi_infer
import numpy as np
import tensorflow as tf
from transformer.model import Transformer
from transformer_beam.multi_decode_model import MultiDecodeModel
from transformer_beam.multi_position_transfer import MultiPositionTransfer
from utils.gradient_util import clip_gradients
from utils.initialize_util import random_normal_variable_initializer


class BatchTrainTest(tf.keras.Model):
  
  def __init__(self):
    super(BatchTrainTest, self).__init__()
    self.transformer_model = Transformer()
    self.multi_position_transfer = MultiPositionTransfer()
    self.multi_decode_model = MultiDecodeModel(self.transformer_model, self.multi_position_transfer)
    
    if initial_memory_trainable:
      self.initial_mems = tf.Variable(random_normal_variable_initializer([1, 1, d_model]))
    
    self.all_outputs = tf.Variable(random_normal_variable_initializer([oracle_mem_len, 1, d_model]))
    
    self.decay = tf.keras.experimental.CosineDecay(2.5e-4, 100000, alpha=0.004)
    
    self.optimizer = tf.optimizers.Adam(self.decay)
  
  def batch_train(self, origin_sequence, relative_to_part_first, valid_mask, beam_mode):
    ''' all these are numpy arrays of shape: [seq_len, batch_size] '''
    ori_sequence = origin_sequence[0:-1,:]
    tgt_sequence = origin_sequence[1:,:]
    seq_len = np.shape(ori_sequence)[0]
    batch_size = np.shape(ori_sequence)[1]
    
    if initial_memory_trainable:
      mems = [tf.tile(self.initial_mems, [1, batch_size, 1]) for _ in range(n_layer)]
    else:
      mems = [tf.zeros([oracle_mem_len, batch_size, d_model], dtype=float_type) for _ in range(n_layer)]
    
    all_outputs = tf.tile(self.all_outputs, [1, batch_size, 1])
    
    i = 0
    while (i < seq_len):
      i_end = i+oracle_mem_len
      i_end = min([seq_len, i_end])
      part_ori_sequence = ori_sequence[i:i_end,:]
      part_tgt_sequence = tgt_sequence[i:i_end,:]
      part_relative_to_part_first = relative_to_part_first[i:i_end,:]
      part_valid_mask = valid_mask[i:i_end,:]
      if beam_mode == multi_infer:
        _, _, _, loss, new_mems = self.multi_decode_model.multi_decode(part_ori_sequence, part_tgt_sequence, part_relative_to_part_first, all_outputs, mems, is_training=True)
      else:
        _, _, _, loss, new_mems = self.transformer_model.transformer(part_ori_sequence, part_tgt_sequence, mems, part_valid_mask, is_training=True)
      mems = new_mems
      i = i_end
      
      with tf.GradientTape() as tape:
        grads = tape.gradient(loss, self.trainable_variables)
        grads = clip_gradients(grads)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
  
  def batch_test(self, origin_sequence, seq_part_skip):
    ''' all these are numpy arrays of shape: [seq_len, batch_size] '''
    
    pass




