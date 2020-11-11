import tensorflow as tf
import numpy as np
import os


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
  top_ks_tensors.append(tf.concat([tf.ones([top_ks[i]]), tf.zeros([top_ks[-1]-top_ks[i]])], axis=0))


home_dir = os.path.expanduser('~')
data_dir = home_dir + "/" + "AST_Tensors"
meta_dir = home_dir + "/" + "AST_Metas"

train_tfxl_tfrecord = data_dir + "/" + "train_tfxl.tfrecord"
test_tfxl_tfrecord = data_dir + "/" + "test_tfxl.tfrecord"

model_storage_parent_dir = 'zoot'
model_storage_dir = "zoot_run_info_record"
model_config = "model_config.txt"
model_check_point = "model_check_point"
model_best = "model_best"
turn_info = "turn_info.txt"
turn_txt = "turn.txt"
best_info = "best_info.txt"
best_txt = "best.txt"
train_noavg = "train_noavg.json"
validate_noavg = "valid_noavg.json"
test_noavg = "test_noavg.json"

ignore_restrain_count = 0
restrain_maximum_count = 10
max_train_epoch = 200
valid_epoch_period = 1







