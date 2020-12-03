from meta_info.non_hyper_constant import data_dir, skeleton_pe,\
  skeleton_one, skeleton_e
import json


oracle_tgt_len = 128
oracle_mem_len = 128
oracle_test_mem_len = 128

n_token = -1
n_layer = 3
d_model = 128
d_embed = 128
n_head = 2
d_head = 16
d_inner = 512
dropout = 0.1
dropatt = 0.0
learning_rate=0.00025
multi_infer_num = 25
batch_size = 10

multi_position_transfer_layer = 3

untie_r = 0

accuracy_based_on_whole = 1
# '''
# token type:
# -1: all 
# 0: only skt
# 1: only non-skt token
# '''
# accuracy_filter_based_on_token_type = accuracy_no_filter

initial_memory_trainable = 0

compute_standard_infer = 1
compute_multi_infer = 1
compute_beam = 1
# run_decode_info = standard_infer
skeleton_mode = skeleton_pe

origin_train_file = data_dir + "/" + skeleton_mode + "_train_data.txt"
train_tfxl_tfrecord = data_dir + "/" + skeleton_mode + "_train_tfxl.tfrecord"
origin_test_file = data_dir + "/" + skeleton_mode + "_test_data.txt"
test_tfxl_tfrecord = data_dir + "/" + skeleton_mode + "_test_tfxl.tfrecord"

''' initialize n_token '''
all_token_summary_file = open(data_dir + "/All_token_summary.json", 'r', encoding='UTF-8')
all_token_summary_ts = json.load(all_token_summary_file)
all_token_summary_file.close()
if skeleton_mode == skeleton_one:
  n_token = 1 + all_token_summary_ts["SkeletonHitNum"] + all_token_summary_ts["SkeletonTokenHitNum"]
elif skeleton_mode == skeleton_pe:
  n_token = 1 + all_token_summary_ts["SkeletonPEHitNum"] + all_token_summary_ts["SkeletonTokenHitNum"]
elif skeleton_mode == skeleton_e:
  n_token = 1 + all_token_summary_ts["SkeletonEachHitNum"] + all_token_summary_ts["SkeletonTokenHitNum"]
else:
  assert False
  
assert n_token > -1

print("n_token:" + str(n_token) + "#SkeletonTokenHitNum:" + str(all_token_summary_ts["SkeletonTokenHitNum"]) + "#SktAccordingNum:" + str(n_token - 1 - all_token_summary_ts["SkeletonTokenHitNum"]))




