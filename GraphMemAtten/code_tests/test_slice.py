import tensorflow as tf
from meta_info.non_hyper_constant import float_type


raw = tf.convert_to_tensor([[1],[2],[3],[4],[5]], float_type)

sliced_raw = tf.slice(raw, [0,0], [-1,-1])

''' errors happened! '''

print(sliced_raw)

last = raw[-1:,:]

print(last)

last2 = raw[-1,:]

print(last2)





