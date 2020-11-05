import tensorflow as tf


accuracy_based_on_whole = 1

top_ks = [1, 3, 6, 10]
mrr_max = 10

oracle_tgt_len = 128
oracle_mem_len = 128
oracle_predict_mem_len = 128

n_token = -1
n_layer = 3
d_model = 128
d_embed = 128
n_head = 2
d_head = 16
d_inner = 512
dropout = 0.1
dropatt = 0.0
initializer = tf.compat.v1.initializers.random_normal(stddev=0.02, seed=None)
learning_rate=0.00025

standard_beam = 0
multi_infer = 1
beam_mode = standard_beam

''' TODO: initialize n_token '''


