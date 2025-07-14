#47 copy

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array #땡겨온 이미지를 수치와

import matplotlib.pyplot as plt
import numpy as np

path = '/workspace/TensorJae/Study25/_data/image/me/'

img = load_img(path + 'me.jpeg', target_size = (100,100))

# print(img)
# print(type(img))
#PIL = Python Image Library

# plt.imshow(img)
# plt.show()  

arr = img_to_array(img)
print(type(arr))
print(arr.shape) #(100,100,3)


###3차원 -> 4차원 
# arr = arr.reshape(1,100,100,3)

img = np.expand_dims(arr, axis=0)
print(img.shape)    #(1, 100, 100, 3)


#me 폴더에 데이터를 npy로 저장하겠다

# np.save(path + 'keras47_me.npy', arr=img)

###################여기부터 증폭###################

datagen = ImageDataGenerator(
    rescale  = 1/255.  ,     #0~255 스케일링,
    # horizontal_flip = True,  #수평 뒤집기 <- 데이터 증폭 또는 변환 /상하반전
    # vertical_flip = True, #수직 뒤집기 <- 데이터 증폭 또는 변환
    width_shift_range = 0.1,#평행이동 10%
    # height_shift_range = 0.1,
    rotation_range = 5,
    # zoom_range = 1.2,
    # shear_range = 0.7,  #좌표 하나 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(찌부 만들기)
    # fill_mode = 'nearest'
    
)



it = datagen.flow(img,                 
    batch_size = 1,                 
)

print("===============================================")
print(it)   #<keras.src.preprocessing.image.NumpyArrayIterator object at 0x7fef5bd3a950>
print("===============================================")

# aaa = it.next()     #파이썬 2.0문법
# print(it.next())
# print(aaa.shape)    #(1, 100, 100, 3)

# bbb = next(it)
# print(bbb)
# print(bbb.shape)    #(1, 100, 100, 3)


# #원래는 안됨. 하지만 it가 더 많으면 가능한데, 지금 여기선 아니기 때문
# print(it.next())
# print(it.next())
# print(it.next())


fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5, 5))
# ax = np.array(ax).flatten()  # 안전하게 1D 배열로

for i in range(5):
    batch = next(it)
    # image = batch[0]
    batch = batch.reshape(100,100,3)

    ax[i].imshow(batch)
    ax[i].axis('off')

# plt.tight_layout()
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# img = np.ones((100, 100, 3), dtype=np.uint8) * 127  # 회색 이미지
# plt.imshow(img)
# plt.axis('off')
# plt.title('Test Image')
# plt.show()


from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
import numpy as np
import os
import subprocess

# 경로 설정
path = '/workspace/TensorJae/Study25/_data/image/me/'
img = load_img(path + 'me.jpeg', target_size=(100, 100))
arr = img_to_array(img)
img = np.expand_dims(arr, axis=0)

# 증강기 설정
datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.1,
    rotation_range=5,
)

it = datagen.flow(img, batch_size=1)

# 저장 디렉토리 생성
save_dir = '/workspace/TensorJae/Study25/_data/image/me/'
os.makedirs(save_dir, exist_ok=True)

# 이미지 저장
for i in range(5):
    batch = next(it)
    image = (batch[0] * 255).astype(np.uint8)
    save_path = os.path.join(save_dir, f'augmented_{i}.jpg')
    Image.fromarray(image).save(save_path)
    print(f"✅ 저장 완료: {save_path}")

# VSCode에서 자동 열기 (폴더 기준)
try:
    subprocess.run(["xdg-open", save_dir])
except Exception as e:
    print(f"❗ 자동 열기 실패: {e}")