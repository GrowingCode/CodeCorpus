import os
from meta_info.hyper_parameter import train_tfxl_tfrecord, test_tfxl_tfrecord


if __name__ == '__main__':
  if os.path.exists(train_tfxl_tfrecord):
    os.remove(train_tfxl_tfrecord)
    
  if os.path.exists(test_tfxl_tfrecord):
    os.remove(test_tfxl_tfrecord)
    
    


