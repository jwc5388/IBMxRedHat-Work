import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


system_profiler SPDisplaysDataType



#gpu 서버에서 로컬 폴더로 복사. 

rsync -avz -e "ssh -p 9020" iclabguest1@117.17.102.50:/home/iclabguest1/TensorJae/Study25/ ~/Desktop/FromGPUServer/Study25/