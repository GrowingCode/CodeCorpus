from builtins import len
import os

from meta_info.hyper_parameter import batch_size, train_tfxl_tfrecord, \
  origin_train_file, test_tfxl_tfrecord, origin_test_file
from meta_info.non_hyper_constant import np_int_type, project_size
import numpy as np
import tensorflow as tf


def generate_tfxl_record(origin_filepath, record_filepath):
  
  writer = tf.io.TFRecordWriter(record_filepath)
  
  ''' the examples must have been sorted! '''
  examples = []
  example_max_ele_size = 0
  
  prev_length = -1
  descend = 0
  
  for line in open(origin_filepath, 'r'):
    one_example = line.strip()
    origin_sequence, relative_to_part_first, valid_mask, seq_part_skip, token_type = one_example.split("#")
    int_origin_sequence = [int(id_str) for id_str in origin_sequence.split()]
    int_relative_to_part_first = [int(id_str) for id_str in relative_to_part_first.split()]
    int_valid_mask = [int(id_str) for id_str in valid_mask.split()]
    int_seq_part_skip = [int(id_str) for id_str in seq_part_skip.split()]
    int_token_type = [int(id_str) for id_str in token_type.split()]
    examples.append((int_origin_sequence, int_relative_to_part_first, int_valid_mask, int_seq_part_skip, int_token_type))
    
    curr_length = len(int_origin_sequence)
    if curr_length < prev_length:
      descend = descend + 1
    
    assert len(int_origin_sequence) == len(int_relative_to_part_first)
    assert len(int_relative_to_part_first) == len(int_valid_mask)
    assert len(int_valid_mask) == len(int_seq_part_skip)
    assert len(int_seq_part_skip) == len(int_token_type)
    
    if (example_max_ele_size < len(int_origin_sequence)):
      example_max_ele_size = len(int_origin_sequence)
    
    if len(examples) == batch_size:
      handle_examples(examples, example_max_ele_size, writer)
      example_max_ele_size = 0
      examples.clear()
      
    prev_length = curr_length
    
  assert descend == project_size - 1
  
  if len(examples) > 0:
    handle_examples(examples, example_max_ele_size, writer)
    example_max_ele_size = 0
    examples.clear()
    
  writer.close()


def pad_vector_to_specified_length(vect, length):
  return np.concatenate([np.array(vect), np.zeros(length-len(vect), np_int_type)], axis=0)
  
  
def handle_examples(examples, example_max_ele_size, writer):
  np_batch_origin_sequence = np.zeros([example_max_ele_size, 0], np_int_type)
  np_batch_relative_to_part_first = np.zeros([example_max_ele_size, 0], np_int_type)
  np_batch_valid_mask = np.zeros([example_max_ele_size, 0], np_int_type)
  np_batch_seq_part_skip = np.zeros([example_max_ele_size, 0], np_int_type)
  np_batch_token_type = np.zeros([example_max_ele_size, 0], np_int_type)
  
  for example in examples:
    np_batch_origin_sequence = np.concatenate([np_batch_origin_sequence, np.expand_dims(pad_vector_to_specified_length(example[0], example_max_ele_size), axis=1)], axis=1)
    np_batch_relative_to_part_first = np.concatenate([np_batch_relative_to_part_first, np.expand_dims(pad_vector_to_specified_length(example[1], example_max_ele_size), axis=1)], axis=1)
    np_batch_valid_mask = np.concatenate([np_batch_valid_mask, np.expand_dims(pad_vector_to_specified_length(example[2], example_max_ele_size), axis=1)], axis=1)
    np_batch_seq_part_skip = np.concatenate([np_batch_seq_part_skip, np.expand_dims(pad_vector_to_specified_length(example[3], example_max_ele_size), axis=1)], axis=1)
    np_batch_token_type = np.concatenate([np_batch_token_type, np.expand_dims(pad_vector_to_specified_length(example[4], example_max_ele_size), axis=1)], axis=1)
    
  features={}
  features['origin_sequence'] = tf.train.Feature(bytes_list = tf.train.BytesList(value=[np_batch_origin_sequence.reshape(-1).tostring()]))
  features['origin_sequence_shape'] = tf.train.Feature(bytes_list = tf.train.BytesList(value=[np.asarray(np.shape(np_batch_origin_sequence)).tostring()]))
  features['relative_to_part_first'] = tf.train.Feature(bytes_list = tf.train.BytesList(value=[np_batch_relative_to_part_first.reshape(-1).tostring()]))
  features['relative_to_part_first_shape'] = tf.train.Feature(bytes_list = tf.train.BytesList(value=[np.asarray(np.shape(np_batch_relative_to_part_first)).tostring()]))
  features['valid_mask'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[np_batch_valid_mask.reshape(-1).tostring()]))
  features['valid_mask_shape'] = tf.train.Feature(bytes_list = tf.train.BytesList(value=[np.asarray(np.shape(np_batch_valid_mask)).tostring()]))
  features['seq_part_skip'] = tf.train.Feature(bytes_list = tf.train.BytesList(value=[np_batch_seq_part_skip.reshape(-1).tostring()]))
  features['seq_part_skip_shape'] = tf.train.Feature(bytes_list = tf.train.BytesList(value=[np.asarray(np.shape(np_batch_seq_part_skip)).tostring()]))
  features['token_type'] = tf.train.Feature(bytes_list = tf.train.BytesList(value=[np_batch_token_type.reshape(-1).tostring()]))
  features['token_type_shape'] = tf.train.Feature(bytes_list = tf.train.BytesList(value=[np.asarray(np.shape(np_batch_token_type)).tostring()]))
  
  tf_features = tf.train.Features(feature = features)
  tf_example = tf.train.Example(features = tf_features)
  tf_serialized = tf_example.SerializeToString()
  writer.write(tf_serialized)
  
  
if __name__ == '__main__':
  if not os.path.exists(train_tfxl_tfrecord):
    generate_tfxl_record(origin_train_file, train_tfxl_tfrecord)
    print("tfxl_record " + train_tfxl_tfrecord + " is generated!")
    
  if not os.path.exists(test_tfxl_tfrecord):
    generate_tfxl_record(origin_test_file, test_tfxl_tfrecord)
    print("tfxl_record " + test_tfxl_tfrecord + " is generated!")
  
  

  
  
  






