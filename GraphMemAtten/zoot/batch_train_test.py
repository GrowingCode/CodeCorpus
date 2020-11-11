from meta_info.hyper_parameter import oracle_mem_len, n_layer, d_model, initial_memory_trainable,\
  compute_beam
from meta_info.non_hyper_constant import float_type, \
  multi_infer_test, multi_infer_train, standard_infer_test, standard_infer_train, top_ks,\
  int_type
import numpy as np
import tensorflow as tf
from transformer.model import Transformer
from transformer_beam.multi_decode_model import MultiDecodeModel
from transformer_beam.multi_position_transfer import MultiPositionTransfer
from utils.accuracy_util import compute_batch_top_ks_accuracy
from utils.gradient_util import clip_gradients
from utils.initialize_util import random_normal_variable_initializer
from transformer_beam.one_sequence.one_seq_beam import OneSeqBeam


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
    
    if compute_beam:
      self.one_seq_beam = OneSeqBeam(self.transformer_model, self.multi_position_transfer)
  
  def batch_train_test(self, origin_sequence, relative_to_part_first, valid_mask, decode_mode):
    ''' all these are numpy arrays of shape: [seq_len, batch_size] '''
    ori_sequence = origin_sequence[0:-1,:]
    tgt_sequence = origin_sequence[1:,:]
    seq_len = np.shape(ori_sequence)[0]
    batch_size = np.shape(ori_sequence)[1]
    
    batch_token_loss = tf.constant(0, float_type)
    batch_token_accuracy = [tf.constant(0, float_type) for _ in range(len(top_ks))]
    batch_token_count = tf.constant(0, float_type)
    
    all_outputs = tf.tile(self.all_outputs, [1, batch_size, 1])
    
    mems = self.get_mems(batch_size)
    
    i = 0
    while (i < seq_len):
      i_end = i+oracle_mem_len
      i_end = min([seq_len, i_end])
      part_ori_sequence = ori_sequence[i:i_end,:]
      part_tgt_sequence = tgt_sequence[i:i_end,:]
      part_relative_to_part_first = relative_to_part_first[i:i_end,:]
      part_valid_mask = valid_mask[i:i_end,:]
      if decode_mode == multi_infer_train or decode_mode == multi_infer_test:
        _, _, predictions, loss, new_mems = self.multi_decode_model.multi_decode(part_ori_sequence, part_tgt_sequence, part_relative_to_part_first, all_outputs, mems, is_training=True)
      elif decode_mode == standard_infer_train or decode_mode == standard_infer_test:
        _, _, predictions, loss, new_mems = self.transformer_model.transformer(part_ori_sequence, part_tgt_sequence, mems, part_valid_mask, is_training=True)
      else:
        assert False

      standard_infer_train = 1
      multi_infer_train = 2
      standard_infer_test = 3
      multi_infer_test = 4

      mems = new_mems
      i = i_end
      
      batch_token_loss += loss
      if decode_mode == standard_infer_train or decode_mode == multi_infer_train:
        with tf.GradientTape() as tape:
          grads = tape.gradient(loss, self.trainable_variables)
          grads = clip_gradients(grads)
          self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
      elif decode_mode == standard_infer_test or decode_mode == multi_infer_test:
        token_accuracy, token_count = compute_batch_top_ks_accuracy(predictions, tgt_sequence, valid_mask)
        temp = [batch_token_accuracy[i]+token_accuracy[i] for i in range(len(top_ks))]
        batch_token_accuracy = temp
        batch_token_count += token_count
      else:
        assert False
    return batch_token_loss, batch_token_accuracy, batch_token_count
  
  def batch_test_beam(self, origin_sequence, seq_part_skip, decode_mode):
    ''' all these are numpy arrays of shape: [seq_len, batch_size] '''
    ''' steps: split origin_sequence to each sequence '''
    sequences = tf.unstack(origin_sequence, axis=1)
    part_skips = tf.unstack(seq_part_skip, axis=1)
    batch_token_each_acc, batch_token_whole_acc, batch_token_count = tf.constant(0, float_type), tf.constant(0, float_type), tf.constant(0, int_type)
    for i in range(len(sequences)):
      sequence = sequences[i]
      part_skip = part_skips[i]
      token_each_acc, token_whole_acc, token_count, _ = self.one_seq_beam(self.get_mems(1), sequence, part_skip, decode_mode)
      batch_token_each_acc += token_each_acc
      batch_token_whole_acc += token_whole_acc
      batch_token_count += token_count
    return batch_token_each_acc, batch_token_whole_acc, batch_token_count
  
  def get_mems(self, batch_size):
    if initial_memory_trainable:
      mems = [tf.tile(self.initial_mems, [1, batch_size, 1]) for _ in range(n_layer)]
    else:
      mems = [tf.zeros([oracle_mem_len, batch_size, d_model], dtype=float_type) for _ in range(n_layer)]
    return mems
    


