from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array #땡겨온 이미지를 수치와

import matplotlib.pyplot as plt
import numpy as np

path = '/workspace/TensorJae/Study25/_data/image/me/'

img = load_img(path + 'me.jpeg', target_size = (100,100))

print(img)
print(type(img))
#PIL = Python Image Library

plt.imshow(img)
plt.show()  

arr = img_to_array(img)
print(type(arr))
print(arr.shape) #(100,100,3)


###3차원 -> 4차원 
# arr = arr.reshape(1,100,100,3)

img = np.expand_dims(arr, axis=0)
print(img.shape)    #(1, 100, 100, 3)


#me 폴더에 데이터를 npy로 저장하겠다

np.save(path + 'keras47_me.npy', arr=img)
