from meta_info.non_hyper_constant import data_dir, skeleton_pe,\
  skeleton_one, skeleton_e
import json


oracle_tgt_len = 128
oracle_mem_len = 128
oracle_test_mem_len = 128

memory_train_test_beam_consistent = 1
additional_filter_memory_when_beam_step_inferring = 1

n_token = -1
n_layer = 6
d_model = 256
d_embed = 256
n_head = 4
d_head = 32
d_inner = 1024
dropout = 0.1
dropatt = 0.0
learning_rate=0.00025
batch_size = 5

multi_infer_num = -1

multi_position_transfer_layer = 1
multi_position_transfer_with_attention_style = 0
multi_position_transfer_atten_head = 256

untie_r = 0

standard_infer_train_to_predict_unk = 0
multi_infer_train_to_predict_unk = 0
accuracy_based_on_whole = 0

initial_memory_trainable = 0

compute_standard_infer = 1
compute_multi_infer = 1
compute_beam = 1

use_simple_multi_infer_mode = 0
skeleton_mode = skeleton_pe

origin_train_file = data_dir + "/" + skeleton_mode + "_train_data.txt"
train_tfxl_tfrecord = data_dir + "/" + skeleton_mode + "_train_tfxl.tfrecord"
origin_test_file = data_dir + "/" + skeleton_mode + "_test_data.txt"
test_tfxl_tfrecord = data_dir + "/" + skeleton_mode + "_test_tfxl.tfrecord"

''' initialize n_skt and n_token '''
all_token_summary_file = open(data_dir + "/All_token_summary.json", 'r', encoding='UTF-8')
all_token_summary_ts = json.load(all_token_summary_file)
all_token_summary_file.close()
all_skt_tensor_summary_file = open(data_dir + "/All_skt_tensor_summary.json", 'r', encoding='UTF-8')
all_skt_tensor_summary_ts = json.load(all_skt_tensor_summary_file)
all_skt_tensor_summary_file.close()
if skeleton_mode == skeleton_one:
  n_skt = all_token_summary_ts["SkeletonHitNum"]
  multi_infer_num = max(all_skt_tensor_summary_ts["max_skt_number_of_one_statement"], all_skt_tensor_summary_ts["max_token_number_of_one_statement"])
elif skeleton_mode == skeleton_pe:
  n_skt = all_token_summary_ts["SkeletonPEHitNum"]
  multi_infer_num = max(all_skt_tensor_summary_ts["max_skt_number_of_pe_statement"], all_skt_tensor_summary_ts["max_token_number_of_pe_statement"])
elif skeleton_mode == skeleton_e:
  n_skt = all_token_summary_ts["SkeletonEachHitNum"]
  multi_infer_num = max(all_skt_tensor_summary_ts["max_skt_number_of_e_statement"], all_skt_tensor_summary_ts["max_token_number_of_e_statement"])
else:
  assert False
  
n_token = n_skt + all_token_summary_ts["SkeletonTokenHitNum"]
assert n_token > -1
multi_infer_num = multi_infer_num + 10

print("n_token:" + str(n_token) + "#multi_infer_num:" + str(multi_infer_num) + "#SkeletonTokenHitNum:" + str(all_token_summary_ts["SkeletonTokenHitNum"]) + "#SktAccordingNum:" + str(n_token - 1 - all_token_summary_ts["SkeletonTokenHitNum"]))







