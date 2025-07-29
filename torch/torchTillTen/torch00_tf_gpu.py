import tensorflow as tf

print('tensorflow version:', tf.__version__)

if tf.config.list_physical_devices('GPU'):
    print('GPU 있음')
    
else:
    print('GPU 없음')
    
#CUDA version
cuda_version = tf.sysconfig.get_build_info()['cuda_version']
print('CUDA version:', cuda_version)

cudnn_version = tf.sysconfig.get_build_info()['cudnn_version']
print('cuDNN version:', cudnn_version)