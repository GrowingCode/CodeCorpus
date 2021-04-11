import tensorflow as tf


msk = tf.bitwise.bitwise_and([1, 0, 0, 1, 1], [0, 1, 1, 0, 1])

print(msk)



