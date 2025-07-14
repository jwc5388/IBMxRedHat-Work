import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd

text = '오늘도 못생기고 영어를 디게 디게 디게 못 하는 이삭이는 재미없는 개그를 \
마구 마구 마구 마구 하면서 딴짓을 한다.'


#Tokenizer는 단어들을 숫자로 매핑하는 준비를 함
#fit_on_texts 는 텍스트를 읽고 어떤 단어가 있는지, 몇번 나오는지 학습
token = Tokenizer()
token.fit_on_texts([text])


#단어 인덱스 확인 - 가장 많이 나온 단어 순서대로 배열
print(token.word_index)

#{'마구': 1, '디게': 2, '오늘도': 3, '못생기고': 4, '영어를': 5, '못': 6, '하는': 7, '이삭이는': 8, '재미없는': 9, '개그를': 10, '하면서': 11, '딴짓을': 12, '한다': 13}

#단어별 출현 횟수
print(token.word_counts)
# OrderedDict([('오늘도', 1), ('못생기고', 1), ('영어를', 1), ('디게', 3), ('못하는', 1), ('이삭이는', 1), ('재미없는', 1), ('개그를', 1), ('마구', 3), 
# ('하면서', 1), ('딴짓을', 1), ('한다', 1)])

x = token.texts_to_sequences([text])
print(x)
#[[3, 4, 5, 2, 2, 2, 6, 7, 8, 9, 10, 1, 1, 1, 1, 11, 12, 13]]

############## 원핫 3가지 맹그러봐 #############
# flattened = x[0]
#1. 판다스
# x = np.array(x[0])        #.flatten() # (1,18) => (18,)
# x = pd.get_dummies(x)
# print(x.shape)


#2. sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
#OneHotEncode는 2D 형태만 받음. 그래서 저기 (-1,1) 형태로 reshape
x= np.array(x[0])
x = x.reshape(-1, 1)
x = encoder.fit_transform(x)
print(x)


#3. keras

# from tensorflow.keras.utils import to_categorical
# x= np.array(x)#.flatten() 
# x = to_categorical(x)
# x = x[:,:,1:]
# x = x.reshape(18,13)
# print(x.shape) # 공칼람 문제 때문에 다르게 만들어줘야함


# 	1.	to_categorical(x)는
# 	•	각 정수(label)를 one-hot vector로 변환.
# 예) 3 → [0,0,0,1,0,…]
# 	2.	x[:,:,1:] 은?
# 	•	간혹 keras to_categorical은 shape이 (batch, timesteps, num_classes) 형태로 나와.
# 	•	여기선 불필요한 첫번째 채널(예: 0 인덱스)을 잘라낸 것처럼 보이는데, 코드 실행 결과 shape 확인 필요.
# 	3.	x.reshape(18,13)
# 	•	18개의 단어가 있고, 고유 단어 인덱스가 13개라면
# 	•	(18,13) shape으로 최종 reshape.

# ⚠️ 주의
# 	•	to_categorical는 class index + 1 차원의 벡터 생성하므로, 인덱스 0이 없는 경우 차원이 하나 더 생길 수 있어.
# 	•	공 칼럼 문제(불필요하게 비어있는 열) 해결을 위해 reshape, slicing이 들어간 듯.

# ⸻
