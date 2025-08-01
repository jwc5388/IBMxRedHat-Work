from re import X
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


augment_size = 100          #증가시킬 사이즈


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)       # (60000,28,28)
print(x_train[0].shape)     #(28,28)

# plt.imshow(x_train[0], cmap = 'gray')

# plt.show()

aaa = np.tile(x_train[0], augment_size).reshape(-1,28,28,1)
print(aaa.shape)
# (100, 28, 28, 1)
datagen = ImageDataGenerator(
    rescale  = 1/255.  ,     #0~255 스케일링,
    # horizontal_flip = True,  #수평 뒤집기 <- 데이터 증폭 또는 변환 /상하반전
    vertical_flip = True, #수직 뒤집기 <- 데이터 증폭 또는 변환
    width_shift_range = 0.1,#평행이동 10%
    # height_shift_range = 0.1,
    rotation_range = 15,
    # zoom_range = 1.2,
    # shear_range = 0.7,  #좌표 하나 고정시키고, 다른 몇개의 좌표를 이동시키는 변환(찌부 만들기)
    fill_mode = 'nearest'
    
)



xy_data = datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),        #x data
    np.zeros(augment_size),     #y data                 
    batch_size = augment_size ,
    shuffle = False,
                     
).next()
#위에 .next 는 첫번쨰 데이터를 뺀것이다.

x_data, y_data = xy_data
print(x_data.shape)#(100, 28, 28, 1)
print(y_data.shape)#(100,)

# print(xy_data)
# print(type(xy_data))    #<class 'tuple'>



print(len(xy_data))     #2
# print(xy_data[0].shape)     #(100, 28, 28, 1)
# print(xy_data[1].shape)     #(100,)


# print(xy_data[0].shape)      #(100, 28, 28, 1)
# print(xy_data[1].shape)      #(100,)

# exit()
# plt.figure(figsize = (7,7))
# for i in range(49):
#     plt.subplot(7,7,i+1)
#     plt.imshow(xy_data[0][i], cmap='gray')
    
# plt.show()


plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(7, 7, i + 1)
    plt.imshow(x_data[i], cmap='gray')
    plt.axis('off')

plt.tight_layout()

# ✅ 파일명 포함한 경로로 저장
save_path = '/workspace/TensorJae/Study25/_data/preview_grid.png'
plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
print(f"✅ Image saved to: {save_path}")



# ✅ xy_data = datagen.flow(...).next()
# flow(...).next() returns a tuple like:

# xy_data = (x_data, y_data)
# So if you use xy_data directly:

# plt.imshow(xy_data[i], cmap='gray')  # ❌
# This is wrong, because xy_data[i] will give either x_data (if i=0) or y_data (if i=1), not an image.

# ✅ x_data, y_data = xy_data
# This unpacks the tuple into:

# x_data.shape = (100, 28, 28, 1)
# y_data.shape = (100,)
# Then this is correct:

# plt.imshow(x_data[i], cmap='gray')  # ✅ Shows image


# xy_data = flow(...).next()	(x_data, y_data)의 튜플을 반환
# xy_data[0]	이미지 배열 x_data
# xy_data[1]	라벨 배열 y_data
# x_data[i]	i번째 이미지 (28x28x1)