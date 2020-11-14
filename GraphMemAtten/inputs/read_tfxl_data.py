import tensorflow as tf
from meta_info.non_hyper_constant import int_type


def parse_function(example_proto):
  # only accept one input: example_proto which is the serialized example
  dics =  {
            'origin_sequence': tf.io.VarLenFeature(dtype=int_type),
            'origin_sequence_shape': tf.io.FixedLenFeature(shape=(2,), dtype=int_type),
            'relative_to_part_first': tf.io.VarLenFeature(dtype=int_type),
            'relative_to_part_first_shape': tf.io.FixedLenFeature(shape=(2,), dtype=int_type),
            'valid_mask': tf.io.VarLenFeature(dtype=int_type),
            'valid_mask_shape': tf.io.FixedLenFeature(shape=(2,), dtype=int_type),
            'seq_part_skip': tf.io.VarLenFeature(dtype=int_type),
            'seq_part_skip_shape': tf.io.FixedLenFeature(shape=(2,), dtype=int_type),
          }
  
  parsed_example = tf.io.parse_single_example(example_proto, dics)
  parsed_example['origin_sequence'] = tf.io.decode_raw(parsed_example['origin_sequence'], int_type)
  parsed_example['origin_sequence_shape'] = tf.io.decode_raw(parsed_example['origin_sequence_shape'], int_type)
  parsed_example['relative_to_part_first'] = tf.io.decode_raw(parsed_example['relative_to_part_first'], int_type)
  parsed_example['relative_to_part_first_shape'] = tf.io.decode_raw(parsed_example['relative_to_part_first_shape'], int_type)
  parsed_example['valid_mask'] = tf.io.decode_raw(parsed_example['valid_mask'], int_type)
  parsed_example['valid_mask_shape'] = tf.io.decode_raw(parsed_example['valid_mask_shape'], int_type)
  parsed_example['seq_part_skip'] = tf.io.decode_raw(parsed_example['seq_part_skip'], int_type)
  parsed_example['seq_part_skip_shape'] = tf.io.decode_raw(parsed_example['seq_part_skip_shape'], int_type)
  
  parsed_example['origin_sequence'] = tf.reshape(parsed_example['origin_sequence'], parsed_example['origin_sequence_shape'])
  parsed_example['relative_to_part_first'] = tf.reshape(parsed_example['relative_to_part_first'], parsed_example['relative_to_part_first_shape'])
  parsed_example['valid_mask'] = tf.reshape(parsed_example['valid_mask'], parsed_example['valid_mask_shape'])
  parsed_example['seq_part_skip'] = tf.reshape(parsed_example['seq_part_skip'], parsed_example['seq_part_skip_shape'])
  
  return parsed_example
  

def generate_parsed_dataset(filename):
  dataset = tf.data.TFRecordDataset(filename)
  new_dataset = dataset.map(parse_function)
  return new_dataset
  
  
  
  
  
  


