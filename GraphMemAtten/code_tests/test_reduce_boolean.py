import tensorflow as tf
from meta_info.non_hyper_constant import bool_type


tensor1 = tf.constant([[True, True],[True, True]], bool_type)
tensor2 = True
bool_tensor = tf.equal(tensor1, tensor2)
result = tf.reduce_all(bool_tensor)
print("result:" + str(result))






