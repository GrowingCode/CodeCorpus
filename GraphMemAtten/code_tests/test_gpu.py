import tensorflow as tf


tf.debugging.set_log_device_placement(True)

print("\n\nNum GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

with tf.device('/gpu:0'):
  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  c = tf.matmul(a, b)
  c_sum = tf.reduce_sum(c)
  c_sum_np = c_sum.numpy()
  if (c_sum_np > 0):
    print("haha sum result:" + str(c_sum_np))
print(c)






