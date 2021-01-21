import tensorflow as tf
import numpy as np
import os
import json


standard_infer = "standard_infer"
multi_infer = "multi_infer"

skeleton_one = "skt_one"
skeleton_pe = "skt_pe"
skeleton_e = "skt_e"

gradient_clip_abs_range = 1000.0

standard_infer_train = 1
multi_infer_train = 2
standard_infer_test = 3
multi_infer_test = 4

bool_type = tf.bool
float_type = tf.float32
int_type = tf.int32

np_bool_type = np.bool
np_float_type = np.float32
np_int_type = np.int32

uniform_range = 0.1
normal_stddev = 0.02
proj_stddev = 0.01

uniform_initializer = tf.keras.initializers.random_uniform(
          minval=-uniform_range,
          maxval=uniform_range,
          seed=None)
normal_initializer = tf.keras.initializers.random_normal(
          stddev=normal_stddev,
          seed=None)
proj_initializer = tf.keras.initializers.random_normal(
          stddev=0.01,
          seed=None)


top_ks = [1, 3, 6, 10]
mrr_max = 10

top_ks_tensors = []
for i in range(len(top_ks)):
  top_ks_tensors.append(tf.concat([tf.ones([top_ks[i]], int_type), tf.zeros([top_ks[-1]-top_ks[i]], int_type)], axis=0))


home_dir = os.path.expanduser('~')
data_dir = home_dir + "/" + "AST_Tensors"
meta_dir = home_dir + "/" + "AST_Metas"

model_storage_parent_dir = 'zoot'
model_storage_dir = "zoot_run_info_record"
model_config = "model_config.txt"
model_check_point = "model_check_point"
model_best = "model_best"
turn_info = "turn_info.txt"
turn_summary = "turn.txt"
best_info = "best_info.txt"
best_summary = "best.txt"
train_noavg = "train_noavg.json"
validate_noavg = "valid_noavg.json"
test_noavg = "test_noavg.json"

ignore_restrain_count = 0
restrain_maximum_count = 10
max_train_epoch = 200
valid_epoch_period = 1

accuracy_no_filter = -1
accuracy_only_skt_filter = 0
accuracy_non_skt_token_filter = 1

debug_in_test_beam = 0
debug_beam_handle_only_one_first_batch = 0
debug_beam_handle_only_one_first_example_in_batch = 0

''' initialize project_size '''
meta_of_app_handle_file = open(meta_dir + "/meta_of_app_handle.json", 'r', encoding='UTF-8')
meta_of_app_handle_ts = json.load(meta_of_app_handle_file)
meta_of_app_handle_file.close()
project_size = meta_of_app_handle_ts["ProjectSize"]

all_skt_one_to_each_file = open(data_dir + "/All_skt_one_to_each.json", 'r', encoding='UTF-8')
all_skt_one_to_each_ts = json.load(all_skt_one_to_each_file)
all_skt_one_to_each_file.close()
all_skt_one_to_each_base = np.array(all_skt_one_to_each_ts[0])
all_skt_one_to_each_start = np.array(all_skt_one_to_each_ts[1])
all_skt_one_to_each_end = np.array(all_skt_one_to_each_ts[2])

all_skt_one_to_pe_file = open(data_dir + "/All_skt_one_to_pe.json", 'r', encoding='UTF-8')
all_skt_one_to_pe_ts = json.load(all_skt_one_to_pe_file)
all_skt_one_to_pe_file.close()
all_skt_one_to_pe_base = np.array(all_skt_one_to_pe_ts[0])
all_skt_one_to_pe_start = np.array(all_skt_one_to_pe_ts[1])
all_skt_one_to_pe_end = np.array(all_skt_one_to_pe_ts[2])

all_skt_pe_to_each_file = open(data_dir + "/All_skt_pe_to_each.json", 'r', encoding='UTF-8')
all_skt_pe_to_each_ts = json.load(all_skt_pe_to_each_file)
all_skt_pe_to_each_file.close()
all_skt_pe_to_each_base = np.array(all_skt_pe_to_each_ts[0])
all_skt_pe_to_each_start = np.array(all_skt_pe_to_each_ts[1])
all_skt_pe_to_each_end = np.array(all_skt_pe_to_each_ts[2])






