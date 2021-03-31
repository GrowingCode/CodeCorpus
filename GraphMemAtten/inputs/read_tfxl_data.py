import tensorflow as tf
from meta_info.non_hyper_constant import int_type
from meta_info.hyper_parameter import train_tfxl_tfrecord


def parse_function(example_proto):
  # only accept one input: example_proto which is the serialized example
  dics =  {
            'origin_sequence': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'origin_sequence_shape': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'relative_to_part_first': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'relative_to_part_first_shape': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'valid_mask': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'valid_mask_shape': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'parent_hint': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'parent_hint_shape': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'position_hint': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'position_hint_shape': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'seq_part_skip': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'seq_part_skip_shape': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'token_type': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'token_type_shape': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'origin_sequence_exact': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            'origin_sequence_exact_shape': tf.io.FixedLenFeature(shape=(), dtype=tf.string),
          }
  
  parsed_example = tf.io.parse_single_example(example_proto, dics)
  parsed_example['origin_sequence'] = tf.io.decode_raw(parsed_example['origin_sequence'], int_type)
  parsed_example['origin_sequence_shape'] = tf.io.decode_raw(parsed_example['origin_sequence_shape'], int_type)
  parsed_example['relative_to_part_first'] = tf.io.decode_raw(parsed_example['relative_to_part_first'], int_type)
  parsed_example['relative_to_part_first_shape'] = tf.io.decode_raw(parsed_example['relative_to_part_first_shape'], int_type)
  parsed_example['valid_mask'] = tf.io.decode_raw(parsed_example['valid_mask'], int_type)
  parsed_example['valid_mask_shape'] = tf.io.decode_raw(parsed_example['valid_mask_shape'], int_type)
  parsed_example['parent_hint'] = tf.io.decode_raw(parsed_example['parent_hint'], int_type)
  parsed_example['parent_hint_shape'] = tf.io.decode_raw(parsed_example['parent_hint_shape'], int_type)
  parsed_example['position_hint'] = tf.io.decode_raw(parsed_example['position_hint'], int_type)
  parsed_example['position_hint_shape'] = tf.io.decode_raw(parsed_example['position_hint_shape'], int_type)
  parsed_example['seq_part_skip'] = tf.io.decode_raw(parsed_example['seq_part_skip'], int_type)
  parsed_example['seq_part_skip_shape'] = tf.io.decode_raw(parsed_example['seq_part_skip_shape'], int_type)
  parsed_example['token_type'] = tf.io.decode_raw(parsed_example['token_type'], int_type)
  parsed_example['token_type_shape'] = tf.io.decode_raw(parsed_example['token_type_shape'], int_type)
  parsed_example['origin_sequence_exact'] = tf.io.decode_raw(parsed_example['origin_sequence_exact'], int_type)
  parsed_example['origin_sequence_exact_shape'] = tf.io.decode_raw(parsed_example['origin_sequence_exact_shape'], int_type)
  
#   print(parsed_example['origin_sequence'])
#   print(parsed_example['origin_sequence_shape'])
  parsed_example['origin_sequence'] = tf.reshape(parsed_example['origin_sequence'], parsed_example['origin_sequence_shape'])
  parsed_example['relative_to_part_first'] = tf.reshape(parsed_example['relative_to_part_first'], parsed_example['relative_to_part_first_shape'])
  parsed_example['valid_mask'] = tf.reshape(parsed_example['valid_mask'], parsed_example['valid_mask_shape'])
  parsed_example['parent_hint'] = tf.reshape(parsed_example['parent_hint'], parsed_example['parent_hint_shape'])
  parsed_example['position_hint'] = tf.reshape(parsed_example['position_hint'], parsed_example['position_hint_shape'])
  parsed_example['seq_part_skip'] = tf.reshape(parsed_example['seq_part_skip'], parsed_example['seq_part_skip_shape'])
  parsed_example['token_type'] = tf.reshape(parsed_example['token_type'], parsed_example['token_type_shape'])
  parsed_example['origin_sequence_exact'] = tf.reshape(parsed_example['origin_sequence_exact'], parsed_example['origin_sequence_exact_shape'])
  
  return parsed_example
  

def generate_parsed_dataset(filename):
  dataset = tf.data.TFRecordDataset(filename)
  new_dataset = dataset.map(parse_function)
  return new_dataset
  
  
if __name__ == '__main__':
  ds = generate_parsed_dataset(train_tfxl_tfrecord)
  for next_element in ds.take(1):
    print(next_element['origin_sequence'])
  


