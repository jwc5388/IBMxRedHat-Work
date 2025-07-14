import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd

text1 = '오늘도 못생기고 영어를 디게 디게 디게 못 하는 이삭이는 재미없는 개그를 \
마구 마구 마구 마구 하면서 딴짓을 한다.'
text2 = '오늘도 박석사가 자아를 디게 디게 찾아냈다. 상진이는 마구 마구 딴짓을 한다.\
    재현은 못생기고 재미없는 딴짓을 한다.'

token = Tokenizer()
token.fit_on_texts([text1,text2])


print(token.word_index)
# {'마구': 1, '디게': 2, '딴짓을': 3, '한다': 4, '오늘도': 5, '못생기고': 6, '재미없는': 7, '영어를': 8, '못': 9, '하는': 10, '이삭이는': 11, '개그를': 12, '하면서': 13, '박석사
# 가': 14, '자아를': 15, '찾아냈다': 16, '상진이는': 17, '재현은': 18}

print(token.word_counts)
# 가': 14, '자아를': 15, '찾아냈다': 16, '상진이는': 17, '재현은': 18}
# OrderedDict([('오늘도', 2), ('못생기고', 2), ('영어를', 1), ('디게', 5), ('못', 1), ('하는', 1), ('이삭이는', 1), ('재미없는', 2), ('개그를', 1), ('마구', 6), ('하면서', 1), ('딴짓을', 3), ('한다', 3), ('박석사가', 1), ('자아를', 1), ('찾아냈다', 1), ('상진이는', 1), ('재현은', 1)])


x = token.texts_to_sequences([text1,text2])
print(x)
# [[5, 6, 8, 2, 2, 2, 9, 10, 11, 7, 12, 1, 1, 1, 1, 13, 3, 4],
#  [5, 14, 15, 2, 2, 16, 17, 1, 1, 3, 4, 18, 6, 7, 3, 4]]
x1 = np.array(x[0])
x2 = np.array(x[1])
x3 = np.concatenate([x1,x2])
print(x3.shape) #(34,)

#Pandas
x4 = pd.get_dummies(x3)
print(x4.shape)

#sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
x3 = x3.reshape(-1,1)
x5 = encoder.fit_transform(x3)
print(x5.shape)

#keras
from keras.utils import to_categorical
x6 = to_categorical(x3)
x6 = x6[:,1:]
print(x6.shape)