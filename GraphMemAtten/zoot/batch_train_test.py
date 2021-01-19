from meta_info.hyper_parameter import oracle_mem_len, n_layer, d_model, initial_memory_trainable,\
  compute_beam
from meta_info.non_hyper_constant import float_type, \
  multi_infer_test, multi_infer_train, standard_infer_test, standard_infer_train, top_ks,\
  int_type, gradient_clip_abs_range,\
  debug_beam_handle_only_one_first_example_in_batch
import numpy as np
import tensorflow as tf
from transformer.model import Transformer
from utils.accuracy_util import compute_batch_top_ks_accuracy
from utils.initialize_util import random_normal_variable_initializer
from transformer_beam.one_sequence.one_seq_beam import OneSeqBeam
from transformer.multi_position_transfer import MultiPositionTransfer
from transformer.multi_decode_model import MultiDecodeModel


class BatchTrainTest(tf.keras.Model):
  
  def __init__(self):
    super(BatchTrainTest, self).__init__()
    self.transformer_model = Transformer()
    self.multi_position_transfer = MultiPositionTransfer()
    self.multi_decode_model = MultiDecodeModel(self.transformer_model, self.multi_position_transfer)
    
    if initial_memory_trainable:
      self.initial_mems = tf.Variable(random_normal_variable_initializer([1, 1, d_model]))
    
    self.all_outputs = tf.Variable(random_normal_variable_initializer([oracle_mem_len, 1, d_model]))
    
#     self.decay = tf.keras.experimental.CosineDecay(2.5e-4, 100000, alpha=0.004)
    
    self.optimizer = tf.optimizers.Adam()# self.decay
    
    if compute_beam:
      self.one_seq_beam = OneSeqBeam(self.transformer_model, self.multi_decode_model)# self.multi_position_transfer
  
  def batch_train_test(self, origin_sequence, relative_to_part_first, valid_mask, token_type, decode_mode):
    ''' all these are numpy arrays of shape: [seq_len, batch_size] '''
#     print(origin_sequence)
    ori_sequence = origin_sequence[0:-1,:]
#     print("=== split line ===")
#     print(ori_sequence)
    tgt_sequence = origin_sequence[1:,:]
    r_relative_to_part_first = relative_to_part_first[1:,:]
    r_valid_mask = valid_mask[1:,:]
    r_token_type = token_type[1:,:]
    seq_len = np.shape(ori_sequence)[0]
#     print("seq_len:" + str(seq_len))
    batch_size = np.shape(ori_sequence)[1]
    
    batch_token_loss = tf.constant(0, float_type)
    batch_token_accuracy = tf.zeros([len(top_ks)], float_type)
    batch_token_count = tf.constant(0, int_type)
    batch_t0_token_accuracy = tf.zeros([len(top_ks)], float_type)
    batch_t0_token_count = tf.constant(0, int_type)
    batch_t1_token_accuracy = tf.zeros([len(top_ks)], float_type)
    batch_t1_token_count = tf.constant(0, int_type)
    all_outputs = tf.tile(self.all_outputs, [1, batch_size, 1])
    
    mems = self.get_mems(batch_size)
    
    i = 0
    while (i < seq_len):
      i_end = i+oracle_mem_len
      i_end = min([seq_len, i_end])
      part_ori_sequence = ori_sequence[i:i_end,:]
      part_tgt_sequence = tgt_sequence[i:i_end,:]
      part_relative_to_part_first = r_relative_to_part_first[i:i_end,:]
      part_valid_mask = r_valid_mask[i:i_end,:]
      part_token_type = r_token_type[i:i_end,:]
      with tf.device("/gpu:0"):
        if decode_mode == standard_infer_train or decode_mode == multi_infer_train:
            with tf.GradientTape() as tape:
              if decode_mode == multi_infer_train:
                all_outputs, _, predictions, loss, new_mems = self.multi_decode_model.multi_decode(part_ori_sequence, part_tgt_sequence, part_relative_to_part_first, all_outputs, mems, part_valid_mask, is_training=True)
              elif decode_mode == standard_infer_train:
                _, _, predictions, loss, new_mems = self.transformer_model.transformer(part_ori_sequence, part_tgt_sequence, mems, part_valid_mask, is_training=True)
              else:
                assert False
        else:
          assert decode_mode == standard_infer_test or decode_mode == multi_infer_test
          if decode_mode == multi_infer_test:
            all_outputs, _, predictions, loss, new_mems = self.multi_decode_model.multi_decode(part_ori_sequence, part_tgt_sequence, part_relative_to_part_first, all_outputs, mems, part_valid_mask, is_training=False)
          elif decode_mode == standard_infer_test:
            _, _, predictions, loss, new_mems = self.transformer_model.transformer(part_ori_sequence, part_tgt_sequence, mems, part_valid_mask, is_training=False)
          else:
            assert False
#       print("loss:" + str(loss))
      mems = new_mems
      i = i_end
#       print("== executed! ==" + str(i))
      batch_token_loss += loss
#       token_count = tf.reduce_sum(part_valid_mask)
#       batch_token_count += token_count
      if decode_mode == standard_infer_train or decode_mode == multi_infer_train:
        with tf.device("/gpu:0"):
          grads = tape.gradient(loss, self.trainable_variables)
#         
#         print("==== print all vars and shape ====")
#         for (grad, var) in zip(grads, self.trainable_variables):
#           shape_str = str(tf.shape(grad)) if grad is not None else "None"
#           print(str(var.name) + " shape:" + shape_str + "#var_shape:" + str(var.shape))
#         
        c_grads = [(tf.clip_by_value(grad, -gradient_clip_abs_range, gradient_clip_abs_range)) if grad != None else None for grad in grads]
#         grads = clip_gradients(grads)
#         nc_grads = [grad if grad is not None else tf.zeros_like(var) for (grad, var) in zip(c_grads, self.trainable_variables)]
        applies = []
        for (grad, var) in zip(c_grads, self.trainable_variables):
          if grad is not None:
            applies.append((grad, var))
        self.optimizer.apply_gradients(applies)# zip(nc_grads, self.trainable_variables)
      elif decode_mode == standard_infer_test or decode_mode == multi_infer_test:
        all_acc, all_tokens, t0_acc, t0_tokens, t1_acc, t1_tokens = compute_batch_top_ks_accuracy(predictions, part_tgt_sequence, part_valid_mask, part_token_type)
#         assert all_tokens.nump() == token_count.numpy()
#         for j in range(len(top_ks)):
#           batch_token_accuracy[j] += token_accuracy[j]
        batch_token_accuracy += all_acc
        batch_token_count += all_tokens
        batch_t0_token_accuracy += t0_acc
        batch_t0_token_count += t0_tokens
        batch_t1_token_accuracy += t1_acc
        batch_t1_token_count += t1_tokens
      else:
        assert False
#     numpy_batch_token_accuracy = [one_token_accuracy.numpy() for one_token_accuracy in batch_token_accuracy]
    return batch_token_loss.numpy(), batch_token_accuracy.numpy(), batch_token_count.numpy(), batch_t0_token_accuracy.numpy(), batch_t0_token_count.numpy(), batch_t1_token_accuracy.numpy(), batch_t1_token_count.numpy()
  
  def batch_test_beam(self, origin_sequence, valid_mask, seq_part_skip, token_type, origin_sequence_exact, decode_mode):
    ''' all these are numpy arrays of shape: [seq_len, batch_size] '''
    ''' steps: split origin_sequence to each sequence '''
    sequences = tf.unstack(origin_sequence, axis=1)
    valid_masks = tf.unstack(valid_mask, axis=1)
    part_skips = tf.unstack(seq_part_skip, axis=1)
    pt_types = tf.unstack(token_type, axis=1)
    sequence_exacts = tf.unstack(origin_sequence_exact, axis=1)
    batch_skt_each_acc, batch_skt_whole_acc, batch_skt_count = tf.zeros([len(top_ks)], float_type), tf.zeros([len(top_ks)], float_type), tf.constant(0, int_type)
    batch_token_each_acc, batch_token_whole_acc, batch_token_count = tf.zeros([len(top_ks)], float_type), tf.zeros([len(top_ks)], float_type), tf.constant(0, int_type)
    for i in range(len(sequences)):
      sequence = sequences[i]
      v_mask = valid_masks[i]
      part_skip = part_skips[i]
      pt_type = pt_types[i]
      sequence_exact = sequence_exacts[i]
      skt_each_acc, skt_whole_acc, skt_count, token_each_acc, token_whole_acc, token_count, _ = self.one_seq_beam(self.get_mems(1), sequence, v_mask, part_skip, pt_type, sequence_exact, decode_mode)
      batch_skt_each_acc += skt_each_acc
      batch_skt_whole_acc += skt_whole_acc
      batch_skt_count += skt_count
      batch_token_each_acc += token_each_acc
      batch_token_whole_acc += token_whole_acc
      batch_token_count += token_count
      if debug_beam_handle_only_one_first_example_in_batch:
        break
    return batch_skt_each_acc.numpy(), batch_skt_whole_acc.numpy(), batch_skt_count.numpy(), batch_token_each_acc.numpy(), batch_token_whole_acc.numpy(), batch_token_count.numpy()
  
  def get_mems(self, batch_size):
    if initial_memory_trainable:
      mems = [tf.tile(self.initial_mems, [1, batch_size, 1]) for _ in range(n_layer)]
    else:
      mems = [tf.zeros([oracle_mem_len, batch_size, d_model], dtype=float_type) for _ in range(n_layer)]
    return mems
    


