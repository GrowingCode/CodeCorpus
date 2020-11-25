import tensorflow as tf
from meta_info.non_hyper_constant import int_type


oracle_valid_mask = tf.constant([0, 1, 1, 0, 1], int_type)
raw_oracle_computed_en_seq = tf.constant([11, 12, 13, 14, 15], int_type)
raw_computed_en_seqs = tf.constant([[1, 2, 3, 4, 5],[6, 7, 8, 9, 10]], int_type)

print(tf.shape(raw_oracle_computed_en_seq))

positive_idx = tf.where(oracle_valid_mask > 0)
oracle_computed_en_seq = tf.gather(raw_oracle_computed_en_seq, positive_idx)
computed_en_seqs = tf.gather(raw_computed_en_seqs, positive_idx, axis=1)

print(oracle_computed_en_seq)
print(computed_en_seqs)

print("==== split line ====")

a = tf.constant([1, 2, 3, 4, 5, 6, 7, 8], int_type)
b = tf.gather(params=a, indices=[-2, -1])
print(b)


