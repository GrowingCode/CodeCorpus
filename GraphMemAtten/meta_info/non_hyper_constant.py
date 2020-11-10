import tensorflow as tf


gradient_clip_abs_range = 1000.0

no_beam = 0
standard_beam = 1
multi_infer = 2

bool_type = tf.bool
float_type = tf.float32
int_type = tf.int32

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


