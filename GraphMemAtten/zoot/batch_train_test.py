import tensorflow as tf
import numpy as np
from meta_info.hyper_parameter import oracle_mem_len, n_layer, d_model
from transformer.model import Transformer
from meta_info.non_hyper_constant import float_type


class BatchTrainTest():
  
  def __init__(self):
    self.transformer_model = Transformer()
    
    pass
  
  def batch_train(self, origin_sequence, relative_to_part_first, valid_mask):
    ''' all these are numpy arrays of shape: [seq_len, batch_size] '''
    ori_sequence = origin_sequence[0:-1,:]
    tgt_sequence = origin_sequence[1:,:]
    seq_len = np.shape(ori_sequence)[0]
    batch_size = np.shape(ori_sequence)[1]
    
    mems = [np.zeros([oracle_mem_len, batch_size, d_model], dtype=float_type)
          for _ in range(n_layer)]
    
    i = 0
    while (i < seq_len):
      i_end = i+oracle_mem_len
      i_end = min([seq_len, i_end])
      part_ori_sequence = ori_sequence[i:i_end,:]
      part_tgt_sequence = tgt_sequence[i:i_end,:]
      part_relative_to_part_first = relative_to_part_first[i:i_end,:]
      part_valid_mask = valid_mask[i:i_end,:]
      _, _, _, loss, new_mems = self.transformer_model.transformer(part_ori_sequence, part_tgt_sequence, mems, part_valid_mask, is_training=True)
      pass
      mems = new_mems
      i = i_end
    pass
  
  def batch_test(self, origin_sequence, seq_part_skip):
    ''' all these are numpy arrays of shape: [seq_len, batch_size] '''
    
    pass




