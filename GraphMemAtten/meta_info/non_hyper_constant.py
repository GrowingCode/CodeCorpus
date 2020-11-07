import tensorflow as tf


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
normal_initializer = tf.compat.v1.initializers.random_normal(
          stddev=normal_stddev,
          seed=None)
proj_initializer = tf.compat.v1.initializers.random_normal(
          stddev=0.01,
          seed=None)

