import json
import os
import time

from inputs.read_tfxl_data import generate_parsed_dataset
from meta_info.hyper_parameter import compute_beam,\
  train_tfxl_tfrecord, origin_train_file, test_tfxl_tfrecord,\
  origin_test_file, compute_multi_infer
from meta_info.non_hyper_constant import model_storage_dir, model_storage_parent_dir, meta_dir, turn_info, \
  model_check_point, turn_summary, best_info, best_summary, model_best, model_config, \
  restrain_maximum_count, max_train_epoch, standard_infer_train, \
  standard_infer_test, multi_infer_test, multi_infer_train, valid_epoch_period, \
  top_ks, np_float_type, standard_infer, multi_infer,\
  debug_beam_handle_only_one_first_batch
import numpy as np
from utils.file_util import copy_files_from_one_directory_to_another_directory
from zoot.batch_train_test import BatchTrainTest
from inputs.write_tfxl_data import generate_tfxl_record


class ModelRunner():
  
  def __init__(self):
    '''
    load training data
    '''
#     data_dir + "/" + "tree_train_data.txt", 
    self.train_ds = generate_parsed_dataset(train_tfxl_tfrecord)
    '''
    load test data
    '''
#     data_dir + "/" + "tree_test_data.txt", 
    self.test_ds = generate_parsed_dataset(test_tfxl_tfrecord)
    '''
    initialize the directory to put the stored model
    '''
    self.real_model_storage_dir = '../' + model_storage_parent_dir + '/' + model_storage_dir
    if not os.path.exists(self.real_model_storage_dir):
      os.makedirs(self.real_model_storage_dir)
    ast_meta_info_dir = self.real_model_storage_dir + "/" + 'data_meta_info'
    if not os.path.exists(ast_meta_info_dir):
      os.makedirs(ast_meta_info_dir)
    copy_files_from_one_directory_to_another_directory(meta_dir, ast_meta_info_dir)
    
    self.config_txt = self.real_model_storage_dir + '/' + model_config
    
    self.batch_train_test_model = BatchTrainTest()
    
#     '''
#     files to store each token accuracy or each token atom accuracy data
#     '''
    
#     self.train_noavg_json = real_model_storage_dir + '/' + train_noavg
#     self.test_noavg_json = real_model_storage_dir + '/' + test_noavg
#     '''
#     set up necessary data
#     '''
#     self.sess = sess
#     place_holders = self.build_input_place_holder()
#     self.build_model_logic()
#     self.optimizer = tf.compat.v1.train.AdamOptimizer()
#     '''
#     build graph of logic 
#     '''
#     self.test_metrics = self.model(place_holders, training = False)
#     ensure_tensor_array_to_tensor_list_in_metrics(self.test_metrics, self.model.metrics_meta, self.model.metrics_index)
#     assert isinstance(self.test_metrics, list)
#     self.test_metrics[-1] = convert_tensor_array_to_lists_of_tensors(make_sure_shape_of_tensor_array(self.test_metrics[-1]))
#     with tf.device('/GPU:0'):
#     self.train_metrics = self.model(place_holders, training = True)
#     ensure_tensor_array_to_tensor_list_in_metrics(self.train_metrics, self.model.metrics_meta, self.model.metrics_index)
#     self.train_metrics[-1] = tf.constant(0, int_type)
#     with tf.GradientTape() as tape:
#       metrics = model(np_array[0], np_array[1], np_array[2], training = training)
#       grads = tape.gradient(metrics[model.metrics_index["all_loss"]], model.trainable_variables)
#       grads = clip_gradients(grads)
#       self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     gvs = self.optimizer.compute_gradients(self.train_metrics[self.model.metrics_index["all_loss"]], tf.compat.v1.trainable_variables(), colocate_gradients_with_ops=True)
#     final_grads = []
#     for (gv, var) in gvs:
#       if gv is not None:
#         grad = tf.clip_by_value(gv, -gradient_clip_abs_range, gradient_clip_abs_range)
#         final_grads.append((grad, var))
#     self.train_op = self.optimizer.apply_gradients(final_grads)
  
  def train_and_test(self, decode_info):
    
    if decode_info == standard_infer:
      train_mode = standard_infer_train
      test_mode = standard_infer_test
    elif decode_info == multi_infer:
      train_mode = multi_infer_train
      test_mode = multi_infer_test
    else:
      assert False
      
    turn_info_txt = self.real_model_storage_dir + '/' + decode_info + "_" + turn_info
    turn_txt = self.real_model_storage_dir + '/' + decode_info + "_" + turn_summary
    check_point_directory = self.real_model_storage_dir + '/' + decode_info + "_" + model_check_point
    check_point_file = check_point_directory + '/' + decode_info + "_" + 'model_weights'
    best_info_txt = self.real_model_storage_dir + '/' + decode_info + "_" + best_info
#     best_txt = self.real_model_storage_dir + '/' + decode_info + "_" + best_summary
    best_model_directory = self.real_model_storage_dir + '/' + decode_info + "_" + model_best
    best_model_file = best_model_directory + '/' + decode_info + "_" + 'model_weights'
    
    turn = 0
    min_loss = None
    max_accuracy = None
    restrain_count = 0
    turn_infos = []
    if os.path.exists(turn_txt):
      assert os.path.exists(turn_info_txt)
      with open(turn_txt, 'r') as turn_record:
        turn_record_json = json.load(turn_record)
        turn = turn_record_json["turn"]
        min_loss = None if "min_loss" not in turn_record_json is None else turn_record_json["min_loss"]
        max_accuracy = None if "max_accuracy" not in turn_record_json is None else turn_record_json["max_accuracy"]
        restrain_count = turn_record_json["restrain_count"]
      with open(turn_info_txt, 'r') as turn_info_record:
        turn_info_lines = turn_info_record.readlines()
        for line in turn_info_lines:
          turn_infos.append(line)
    '''
    restore model when turn is not 0
    '''
    
    if restrain_count >= restrain_maximum_count:
      turn = max_train_epoch
    if turn > 0 and turn < max_train_epoch:
      self.batch_train_test_model.load_weights(check_point_file)
    '''
    begin real training procedure
    '''
    total_turn_sum_train_time_cost = 0.0
    total_turn_sum = 0.0
    total_turn_average_train_time_cost = 0.0
    while turn < max_train_epoch:
      '''
      one epoch starts
      '''
      if turn > 0:
        train_start_time = time.time()
        train_output_result = model_running(self.batch_train_test_model, self.train_ds, train_mode)
        train_end_time = time.time()
        train_time_cost = train_end_time - train_start_time
        total_turn_sum_train_time_cost = total_turn_sum_train_time_cost + train_time_cost
        total_turn_sum = total_turn_sum + 1.0
        total_turn_average_train_time_cost = total_turn_sum_train_time_cost / total_turn_sum
        ''' exactly record train loss '''
  #       print("train_output_result:" + str(train_output_result))
        train_avg = compute_average(train_output_result)
  #       print("train_avg:" + str(train_avg))
  #       train_noavg = process_noavg(train_output_result)
  #       with open(self.train_noavg_json, 'w') as train_noavg_record:
  #         train_noavg_record.write(json.dumps(train_noavg))
        train_average_loss = train_avg["average_token_loss"]
        '''
        compute average loss when training
        '''
        print(str(turn+1) + "/" + str(max_train_epoch) + " turn's train_set_average_loss:" + str(train_average_loss))
      else:
        print("turn:" + str(turn) + " do not train.")
      '''
      testing process if period reached
      '''
      train_compute_valid = (turn+1) % valid_epoch_period == 0
      if train_compute_valid:
#         print("== running before test ==")
        valid_output_result = model_running(self.batch_train_test_model, self.train_ds, test_mode)
#         print("== running after test ==")
        valid_avg = compute_average(valid_output_result)
#         valid_noavg = process_noavg(valid_output_result)
        '''
        compute average loss
        '''
        valid_average_loss = 0.0
        valid_average_accuracy = np.zeros([len(top_ks)], dtype=np_float_type).tolist()
        if valid_avg:
          valid_average_loss = valid_avg["average_token_loss"]
          valid_average_accuracy = valid_avg["average_token_accuracy"]
          print(str(turn+1) + "/" + str(max_train_epoch) + " turn" + "#" + json.dumps(valid_avg))
        '''
        save best model
        '''
        to_save_best_model = False
        if max_accuracy is None:
          max_accuracy = valid_average_accuracy
          min_loss = valid_average_loss
          to_save_best_model = True
        else:
          if newer_accuracy_is_better(max_accuracy, valid_average_accuracy):
            print("max_accuracy[0]:" + str(max_accuracy[0]) + "#valid_average_accuracy[0]:" + str(valid_average_accuracy[0]))
            if round(max_accuracy[0], 6) == round(valid_average_accuracy[0], 6):
              restrain_count = restrain_count + 1
              print("restrain_count:" + str(restrain_count))
            else:
              restrain_count = 0
            max_accuracy = valid_average_accuracy
            min_loss = valid_average_loss
            to_save_best_model = True
          else:
            restrain_count = restrain_count + 1
        if to_save_best_model:
#           tf.compat.v1.train.Saver().save(self.sess, self.best_model_file)
          self.batch_train_test_model.save_weights(best_model_file)
          with open(best_info_txt, 'w') as best_info_record:
            best_info_record.write("the_turn_generating_best_model:" + str(turn+1) + "#" + dict_to_string(valid_avg))
#           with open(self.valid_noavg_json, 'w') as valid_noavg_record:
#             valid_noavg_record.write(json.dumps(valid_noavg))
          print("========== Saved best model ==========")
          
        turn_infos.append(dict_to_string(valid_avg))
        '''
        save turn model
        judge whether the model is best currently
        the following if judgment is to decide whether the model has been restrained for a while, if so stop instantly
        save the model if period reached
        save check point model
        '''
#         tf.compat.v1.train.Saver().save(self.sess, self.check_point_file)
        self.batch_train_test_model.save_weights(check_point_file)
        '''
        write the turn to file
        '''
        turn_record_json2 = {}
        with open(turn_txt, 'w') as turn_record:
          turn_record_json2["turn"] = turn+1
          turn_record_json2["restrain_count"] = restrain_count
          turn_record_json2["min_loss"] = min_loss
          turn_record_json2["max_accuracy"] = max_accuracy
          turn_record_json2["average_train_time_cost"] = total_turn_average_train_time_cost
          turn_record.write(json.dumps(turn_record_json2))
        with open(turn_info_txt, 'w') as turn_info_record:
          t_info_record = '\n'.join(turn_infos)
          turn_info_record.write(t_info_record)
        print("========== Saved check point model ==========")
        '''
        go to next epoch
        '''
      if (restrain_count >= restrain_maximum_count):
        turn = max_train_epoch
      turn = turn+1
  
  def test_beam(self, decode_info):
    
    if decode_info == standard_infer:
      test_mode = standard_infer_test
    elif decode_info == multi_infer:
      test_mode = multi_infer_test
    else:
      assert False
    
#     best_info_txt = self.real_model_storage_dir + '/' + decode_info + "_" + best_info
    best_txt = self.real_model_storage_dir + '/' + decode_info + "_" + best_summary
#     '''
#     compute valid set each_noavg accuracy
#     '''
#     if compute_extra_valid_each_noavg:
#       print("===== Validating procedure starts! =====")
#       output_result = self.model_running(validating_mode)
#       avg = compute_average(output_result)
#       noavg = process_noavg(output_result)
#       with open(best_info_txt, 'r') as best_info_record:
#         best_turn_str = get_content_between_two_specified_string(best_info_record.read(), ":", "#")
#       with open(best_info_txt, 'w') as best_info_record:
#         best_info_record.write("the_turn_generating_best_model:" + best_turn_str + "#" + dict_to_string(avg))
#       with open(self.valid_noavg_json, 'w') as valid_noavg_record:
#         valid_noavg_record.write(json.dumps(noavg))
#       print(dict_to_string(avg))
#       print("===== Valid extra procedure is over! =====")
    '''
    compute average loss
    test set loss/accuracy leaves_score/all_score
    '''
    print("===== Testing procedure starts! =====")
    output_result = beam_model_running(self.batch_train_test_model, self.test_ds, test_mode)
    avg = compute_average(output_result)
#     noavg = process_noavg(output_result)
    with open(best_txt, 'w') as best_model_statement_accuracy_record:
      best_model_statement_accuracy_record.write(json.dumps(avg))
#     with open(self.test_noavg_json, 'w') as test_noavg_record:
#       test_noavg_record.write(json.dumps(noavg))
    print(dict_to_string(avg))
  
  def restore_best(self, decode_info):
    best_model_directory = self.real_model_storage_dir + '/' + decode_info + "_" + model_best
    best_model_file = best_model_directory + '/' + decode_info + "_" + 'model_weights'
    
    self.batch_train_test_model.load_weights(best_model_file)
    print("Restored best model in " + best_model_directory)
  
  
def model_running(model, ds, decode_mode):
  all_token_accuracy = np.zeros(len(top_ks), np_float_type)
  all_token_count = 0
  all_token_loss = 0
#   iterator = ds.make_one_shot_iterator()
#   next_element = iterator.get_next()
  start_time = time.time()
  i = 1
#   while True:
#     try:
  for next_element in ds:
    batch_token_loss, batch_token_accuracy, batch_token_count = model.batch_train_test(next_element['origin_sequence'], next_element['relative_to_part_first'], next_element['valid_mask'], decode_mode)
    all_token_loss += batch_token_loss
    all_token_accuracy += np.asarray(batch_token_accuracy, np_float_type)
    all_token_count += batch_token_count
#     except tf.errors.OutOfRangeError:
#       break
#     else:
#       assert False
    i+=1
  end_time = time.time()
#   print("all_token_loss:" + str(all_token_loss))
  print("mode:" + str(decode_mode) + "#batch_size:" + str(i) + "#time_cost:" + str(round(end_time-start_time, 1)) +"s")
  return {'token_loss':all_token_loss, 'token_accuracy':all_token_accuracy, 'token_count':all_token_count}
  
  
def beam_model_running(model, ds, decode_mode):
  all_token_each_acc = 0
  all_token_whole_accuracy = 0
  all_token_count = 0
#   iterator = ds.make_one_shot_iterator()
#   next_element = iterator.get_next()
  start_time = time.time()
  i = 1
#   while True:
#     try:
  r_ds = ds
  if debug_beam_handle_only_one_first_batch:
    r_ds = ds.take(1)
  for next_element in r_ds:
    batch_token_each_acc, batch_token_whole_acc, batch_token_count = model.batch_test_beam(next_element['origin_sequence'], next_element['valid_mask'], next_element['seq_part_skip'], decode_mode)
    all_token_each_acc += batch_token_each_acc
    all_token_whole_accuracy += batch_token_whole_acc
    all_token_count += batch_token_count
#     except tf.errors.OutOfRangeError:
#       break
#     else:
#       assert False
    i+=1
  end_time = time.time()
  print("mode:" + str(decode_mode) + "#batch_size:" + str(i) + "#time_cost:" + str(round(end_time-start_time, 1)) +"s")
  return {'token_each_acc':all_token_each_acc, 'token_whole_accuracy':all_token_whole_accuracy, 'token_count':all_token_count}


#   def model_running_one_example(self, training, one_example):
#     feed_dict = self.build_feed_dict(one_example)
#     if training:
#       r_metrics = self.sess.run([self.train_op, *self.train_metrics], feed_dict=feed_dict)
#       metrics = r_metrics[1:]
#       filter_out_invalid_mark_in_metrics(metrics, self.model.metrics_meta, self.model.metrics_index)
#     else:
#       metrics = self.sess.run([*self.test_metrics], feed_dict=feed_dict)
#       filter_out_invalid_mark_in_metrics(metrics, self.model.metrics_meta, self.model.metrics_index)
#     return metrics


# def merge_metric(all_metrics, part_metric):
#   ''' assert two structures are same '''
#   all_metrics_is_empty = True
#   if all_metrics:
#     all_metrics_is_empty = False
#   for key in part_metric:
#     if not all_metrics_is_empty:
#       assert key in all_metrics
#     p_one_item = part_metric[key]
#     if type(p_one_item) == dict:
#       if all_metrics_is_empty:
#         all_metrics[key] = {}
#       merge_metric(all_metrics[key], p_one_item)
#     else:
#       if key.endswith("_noavg"):
#         if all_metrics_is_empty:
#           all_metrics[key] = []
#         all_metrics[key].append(p_one_item)
#       else:
#         if all_metrics_is_empty:
#           all_metrics[key] = p_one_item
#         else:
#           all_metrics[key] = all_metrics[key] + p_one_item
# 
# 
def compute_average(dict_t):
  r = {}
  for k in dict_t:
    if not k.endswith('_count') and not k.endswith('_noavg'):
      idx = k.find('_')
      assert idx > 0
      first_k_part = k[0:idx]
      k_count = first_k_part + '_count'
      k_tm = "average_" + k
      divd = dict_t[k_count].astype(np_float_type)
      if divd == 0.0:
        divd = 0.0000000001
      r[k_tm] = de_numpy(dict_t[k]/divd)
    elif k.endswith('_count'):
      r[k] = de_numpy(dict_t[k])
  return r


# def process_noavg(dict_t):
#   r = {}
#   for k in dict_t:
#     if k.endswith('_noavg'):
#       r[k] = de_numpy(dict_t[k])
#   return r


def de_numpy(d):
  if isinstance(d, (np.ndarray)):
    return d.tolist()
  elif isinstance(d, (list, tuple)):
    r = []
    for ele in d:
      r.append(de_numpy(ele))
    return r;
  else:
    if isinstance(d, (np.float16, np.float32, np.float64)):
      return float(d)
    if isinstance(d, (np.int8, np.int16, np.int32, np.int64)):
      return int(d)


def dict_to_string(dict_t):
  r = ""
  for k in dict_t:
    t = dict_t[k]
    if (type(t) == dict):
      r = r + dict_to_string(t)
    else:
      r = r + "#" + k + ":" + str(t)
  return r


def newer_accuracy_is_better(old_accuracy, new_accuracy):
  for i in range(len(top_ks)):
    if old_accuracy[i] < new_accuracy[i]:
      return True
    elif old_accuracy[i] > new_accuracy[i]:
      return False


def info_of_train_stop_test_start(average_accuracy):
  print("===== Because training accuracy is very high:" + str(average_accuracy) + ", training procedure will stop soon! =====")
  time.sleep(1)
  

# def model_output(tensors, tensors_meta):
#   assert len(tensors) == len(tensors_meta), "tensors length:" + str(len(tensors)) + "#tensors_meta length:" + str(len(tensors_meta))
#   numpy_dict = {}
#   for i, t in enumerate(tensors):
#     numpy_dict[tensors_meta[i][0]] = t
#   return numpy_dict
  

if __name__ == '__main__':
  if not os.path.exists(train_tfxl_tfrecord):
    generate_tfxl_record(origin_train_file, train_tfxl_tfrecord)
    
  if not os.path.exists(test_tfxl_tfrecord):
    generate_tfxl_record(origin_test_file, test_tfxl_tfrecord)
  
  mr = ModelRunner()
  mr.train_and_test(standard_infer)
  mr.restore_best(standard_infer)
  if compute_beam:
    mr.test_beam(standard_infer)
  
  if compute_multi_infer:
    mr.train_and_test(multi_infer)
    mr.restore_best(multi_infer)
    if compute_beam:
      mr.test_beam(multi_infer)













  

