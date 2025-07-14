# import numpy as np
# from sklearn.datasets import fetch_california_housing, load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import r2_score
# import warnings
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
# warnings.filterwarnings('ignore')
# import signal


# from sklearn.utils import all_estimators
# import sklearn as sk
# print(sk.__version__)  # 1.6.1


# # 1. Load Data
# path = './Study25/_data/kaggle/bank/'

# train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# # 2. Encode categorical features
# le_geo = LabelEncoder()
# le_gender = LabelEncoder()

# train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
# train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])

# test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
# test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# # 3. Drop unneeded columns
# train_csv = train_csv.drop(['CustomerId', 'Surname'], axis=1)
# test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# # 4. Separate features and target
# x = train_csv.drop(['Exited'], axis=1)
# y = train_csv['Exited']


# # ✅ Add scaler here
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# test_csv = scaler.transform(test_csv)


# # 6. Train-test split (after scaling)
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, random_state=33
# )


# # model = RandomForestRegressor()
# allAlgorithms = all_estimators(type_filter='classifier')
# print('allAlgoriths: ', allAlgorithms)


# print('모델의 갯수:', len(allAlgorithms))   #55
# print(type(allAlgorithms))  #<class 'list'>

# # 타임아웃 설정 함수
# class TimeoutException(Exception):
#     pass

# def handler(signum, frame):
#     raise TimeoutException()

# signal.signal(signal.SIGALRM, handler)  # SIGALRM 시그널 핸들러 설정

# max_score = 0

# for name, algorithm in allAlgorithms:
#     try:
#         model = algorithm()
#         signal.alarm(10)  # ⏱️ 최대 10초 제한

#         model.fit(x_train, y_train)
#         result = model.score(x_test, y_test)

#         signal.alarm(0)  # 성공 시 알람 종료

#         if result > max_score:
#             max_score = result
#             max_name = name

#         print(f'{name} 의 정답률 : {result:.4f}')

#     except TimeoutException:
#         print(f'{name} 은(는) ⏱️ 시간 초과!')
#     except Exception as e:
#         print(f'{name} 은(는) 에러 발생: {e}')
#         signal.alarm(0)  # 예외 발생 시에도 알람 종료는 꼭 해야 함

# print('======================================================')
# print('최고모델:', max_name, max_score)
# print('======================================================')


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
import sklearn as sk
from sklearn.datasets import load_wine

print(sk.__version__) #1.6.1


###############################################################################
#kaggle bank data
# #1. 데이터
# path = '/Users/jaewoo000/Desktop/IBM:Redhat/Study25/_data/kaggle/bank/'

# train_csv = pd.read_csv(path + 'train.csv',index_col=0)
# test_csv = pd.read_csv(path + 'test.csv',index_col=0)
# submission_csv = pd.read_csv(path+'sample_submission.csv',index_col=0)

# le_geo = LabelEncoder() # 클래스를 인스턴스 한다.
# le_gender = LabelEncoder()
# #print(train_csv.columns)
# # Index(['CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender', 'Age',
# #        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
# #        'EstimatedSalary', 'Exited'],

# train_csv['Geography'] = le_geo.fit_transform(train_csv['Geography'])
# train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])
# print(train_csv['Geography'])
# print(train_csv['Geography'].value_counts()) #pandas
# # 0    94215
# # 2    36213
# # 1    34606
# print(train_csv['Gender'].value_counts()) #pandas np.unique(data, return_counts=True)
# # 1    93150
# # 0    71884
# test_csv['Geography'] = le_geo.transform(test_csv['Geography'])
# test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# train_csv = train_csv.drop(['CustomerId','Surname'],axis= 1)
# test_csv = test_csv.drop(['CustomerId','Surname'], axis= 1)
# # print(train_csv.columns)
# # print(test_csv)

# train_csv['Balance'] = train_csv['Balance'].replace(0, train_csv['Balance'].mean())
# test_csv['Balance'] = test_csv['Balance'].replace(0, test_csv['Balance'].mean())
# #train_csv.dropna()

# x = train_csv.drop(['Exited'], axis=1)
# y = train_csv['Exited']

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.1, random_state= 42)


###############################################################################



# 1. Load Data
dataset = load_wine()
x = dataset.data
y = dataset.target  # shape: (178,), classes: 0, 1, 2

# 3. Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42
)


#2. 모델
# model = RandomForestRegressor()
allAlgorithms = all_estimators(type_filter='classifier')


# 모델 이름 출력
print('allAlgorithms : ', allAlgorithms)
print('모델의 갯수 : ', len(allAlgorithms))
print(type(allAlgorithms))

max_name=''
max_score=0
for(name,algorithm) in allAlgorithms:
    try:
        model = algorithm()
        #3. 훈련
        model.fit(x_train, y_train)

        #4 평가 예측
        results = model.score(x_test,y_test)
        print(name, '의 정답률', results)

        if max_score < results :
            max_score = results
            max_name = name
    except:
        print(name,'은(는) 에러뜬 분!!!')

print('========================================')
print('최고모델 : ', max_name, max_score) # 최고모델 :  ExtraTreesClassifier 1.0
print('========================================')
#3. 훈련 컴파일

#4 .평가 예측