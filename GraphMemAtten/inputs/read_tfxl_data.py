import tensorflow as tf
from meta_info.non_hyper_constant import float_type, int_type


def parse_function(example_proto):
  # only accept one input: example_proto which is the serialized example
  dics = {
            'origin_sequence': tf.io.VarLenFeature(dtype=int_type),
            'origin_sequence_shape': tf.io.FixedLenFeature(shape=(2,), dtype=int_type),
            'relative_to_part_first': tf.io.VarLenFeature(dtype=int_type),
            'relative_to_part_first_shape': tf.io.FixedLenFeature(shape=(2,), dtype=int_type),
            'valid_mask': tf.io.VarLenFeature(dtype=int_type),
            'valid_mask_shape': tf.io.FixedLenFeature(shape=(2,), dtype=int_type),
            'seq_part_skip': tf.io.VarLenFeature(dtype=int_type),
            'seq_part_skip_shape': tf.io.FixedLenFeature(shape=(2,), dtype=int_type),
          }
  
  
  
  



  
  
  


