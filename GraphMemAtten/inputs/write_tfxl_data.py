import tensorflow as tf
import numpy as np
from meta_info.non_hyper_constant import data_dir, np_int_type
from meta_info.hyper_parameter import batch_size
from builtins import len


def generate_tfxl_record(mode, record_filename):
  
  writer = tf.io.TFRecordWriter(record_filename)
  
  ''' the examples must have been sorted! '''
  examples = []
  example_max_ele_size = 0
  
  for line in open(data_dir + "/" + "tfxl_" + mode + "_data.txt", 'r'):
    one_example = line.strip()
    origin_sequence, relative_to_part_first, valid_mask, seq_part_skip = one_example.split("#")
    int_origin_sequence = [int(id_str) for id_str in origin_sequence.split()]
    int_relative_to_part_first = [int(id_str) for id_str in relative_to_part_first.split()]
    int_valid_mask = [int(id_str) for id_str in valid_mask.split()]
    int_seq_part_skip = [int(id_str) for id_str in seq_part_skip.split()]
    examples.append((int_origin_sequence, int_relative_to_part_first, int_valid_mask, int_seq_part_skip))
    
    assert len(int_origin_sequence) == len(int_relative_to_part_first)
    assert len(int_relative_to_part_first) == len(int_valid_mask)
    assert len(int_valid_mask) == len(int_seq_part_skip)
    
    if (example_max_ele_size < len(int_origin_sequence)):
      example_max_ele_size = len(int_origin_sequence)
    
    if len(examples) == batch_size:
      handle_examples(examples, example_max_ele_size, writer)
      example_max_ele_size = 0
      examples.clear()
      
  if len(examples) > 0:
    handle_examples(examples, example_max_ele_size, writer)
    example_max_ele_size = 0
    examples.clear()
    
  writer.close()


def pad_vector_to_specified_length(vect, length):
  return np.concatenate([np.array(vect), np.zeros(length-len(vect), np_int_type)], axis=0)
  
  
def handle_examples(examples, example_max_ele_size, writer):
  np_batch_origin_sequence = np.zeros([0, example_max_ele_size], np_int_type)
  np_batch_relative_to_part_first = np.zeros([0, example_max_ele_size], np_int_type)
  np_batch_valid_mask = np.zeros([0, example_max_ele_size], np_int_type)
  np_batch_seq_part_skip = np.zeros([0, example_max_ele_size], np_int_type)
  
  for example in examples:
    np_batch_origin_sequence = np.concatenate([np_batch_origin_sequence, np.expand_dims(pad_vector_to_specified_length(example[0], example_max_ele_size), axis=0)], axis=0)
    np_batch_relative_to_part_first = np.concatenate([np_batch_relative_to_part_first, np.expand_dims(pad_vector_to_specified_length(example[1], example_max_ele_size), axis=0)], axis=0)
    np_batch_valid_mask = np.concatenate([np_batch_valid_mask, np.expand_dims(pad_vector_to_specified_length(example[2], example_max_ele_size), axis=0)], axis=0)
    np_batch_seq_part_skip = np.concatenate([np_batch_seq_part_skip, np.expand_dims(pad_vector_to_specified_length(example[3], example_max_ele_size), axis=0)], axis=0)
    
  features={}
  features['origin_sequence'] = tf.train.Feature(int64_list=tf.train.BytesList(np_batch_origin_sequence.reshape(-1)))
  features['origin_sequence_shape'] = tf.train.Feature(float_list = tf.train.BytesList(np.shape(np_batch_origin_sequence)))
  features['relative_to_part_first'] = tf.train.Feature(int64_list = tf.train.BytesList(value=np_batch_relative_to_part_first.reshape(-1)))
  features['relative_to_part_first_shape'] = tf.train.Feature(float_list = tf.train.BytesList(np.shape(np_batch_relative_to_part_first)))
  features['valid_mask'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=np_batch_valid_mask.reshape(-1)))
  features['valid_mask_shape'] = tf.train.Feature(int64_list = tf.train.BytesList(np.shape(np_batch_valid_mask)))
  features['seq_part_skip'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=np_batch_seq_part_skip.reshape(-1)))
  features['seq_part_skip_shape'] = tf.train.Feature(int64_list = tf.train.BytesList(np.shape(np_batch_seq_part_skip)))
  
  tf_features = tf.train.Features(feature = features)
  tf_example = tf.train.Example(features = tf_features)
  tf_serialized = tf_example.SerializeToString()
  writer.write(tf_serialized)
  
  
  
  





