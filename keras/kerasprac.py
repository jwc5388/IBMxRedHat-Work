# # # # # # # # # # # # # # # # # # # # # import tensorflow as tf
# # # # # # # # # # # # # # # # # # # # # import numpy as np

# # # # # # # # # # # # # # # # # # # # # from tensorflow.keras.models import Sequential
# # # # # # # # # # # # # # # # # # # # # from tensorflow.keras.layers import Dense


# # # # # # # # # # # # # # # # # # # # # x = np.array([1,2,3,4,5,6,7,8,9,10]) 
# # # # # # # # # # # # # # # # # # # # # y = np.array([2,4,6,8,10,12,14,16,18,20]) 

# # # # # # # # # # # # # # # # # # # # # model = Sequential()
# # # # # # # # # # # # # # # # # # # # # model.add(Dense(1, input_dim=1))

# # # # # # # # # # # # # # # # # # # # # #Mean Squared Error (MSE), which is a common loss function used for regression tasks. 
# # # # # # # # # # # # # # # # # # # # # # It calculates the average squared difference between the predicted values and the actual values. 
# # # # # # # # # # # # # # # # # # # # # # The model tries to minimize this error during training.

# # # # # # # # # # # # # # # # # # # # # # Optimizers are algorithms that adjust the weights of the neural network to minimize the loss function. 
# # # # # # # # # # # # # # # # # # # # # # In this case, Adam (short for Adaptive Moment Estimation) is used.
# # # # # # # # # # # # # # # # # # # # # model.compile(loss='mse', optimizer = 'adam')

# # # # # # # # # # # # # # # # # # # # # #x is input data, y is target data
# # # # # # # # # # # # # # # # # # # # # #epochs = number of times the whole dataset is passed into the network during training
# # # # # # # # # # # # # # # # # # # # # model.fit(x,y, epochs=11000)

# # # # # # # # # # # # # # # # # # # # # result = model.predict(np.array([30]))

# # # # # # # # # # # # # # # # # # # # # print('prediction of 20: ' , result)



# # # # # # # # # # # # # # # # # # # # # import sklearn as sk
# # # # # # # # # # # # # # # # # # # # # from sklearn.datasets import load_boston
# # # # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # # # from tensorflow.keras.models import Sequential
# # # # # # # # # # # # # # # # # # # # # from tensorflow.keras.layers import Dense
# # # # # # # # # # # # # # # # # # # # # from sklearn.model_selection import train_test_split

# # # # # # # # # # # # # # # # # # # # # #1 data
# # # # # # # # # # # # # # # # # # # # # dataset = load_boston()
# # # # # # # # # # # # # # # # # # # # # #Describe
# # # # # # # # # # # # # # # # # # # # # print(dataset.DESCR)
# # # # # # # # # # # # # # # # # # # # # print(dataset.feature_names)

# # # # # # # # # # # # # # # # # # # # # x = dataset.data
# # # # # # # # # # # # # # # # # # # # # y = dataset.target

# # # # # # # # # # # # # # # # # # # # import sklearn as sk

# # # # # # # # # # # # # # # # # # # # from sklearn.datasets import load_boston
# # # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # # from tensorflow.keras.models import Sequential
# # # # # # # # # # # # # # # # # # # # from tensorflow.keras.layers import Dense
# # # # # # # # # # # # # # # # # # # # from sklearn.model_selection import train_test_split

# # # # # # # # # # # # # # # # # # # # dataset = load_boston()
# # # # # # # # # # # # # # # # # # # # #Describe 
# # # # # # # # # # # # # # # # # # # # print(dataset.DESCR)
# # # # # # # # # # # # # # # # # # # # print(dataset.feature_names)

# # # # # # # # # # # # # # # # # # # # x = dataset.data
# # # # # # # # # # # # # # # # # # # # y = dataset.target

# # # # # # # # # # # # # # # # # # # # print(x.shape)

# # # # # # # # # # # # # # # # # # # # x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state=3)


# # # # # # # # # # # # # # # # # # # # #2 model
# # # # # # # # # # # # # # # # # # # # model = Sequential()
# # # # # # # # # # # # # # # # # # # # model.add(Dense(10, input_dim = 13))
# # # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # # model.add(Dense(1))


# # # # # # # # # # # # # # # # # # # # #3 compile and train
# # # # # # # # # # # # # # # # # # # # model.compile(loss = 'mse', optimizer = 'adam')
# # # # # # # # # # # # # # # # # # # # model.fit(x_train, y_train, epochs = 200, batch_size = 1)

# # # # # # # # # # # # # # # # # # # # loss = model.evaluate(x_test,y_test)
# # # # # # # # # # # # # # # # # # # # result = model.predict(x_test)

# # # # # # # # # # # # # # # # # # # # from sklearn.metrics import r2_score

# # # # # # # # # # # # # # # # # # # # r2 = r2_score(y_test, result)
# # # # # # # # # # # # # # # # # # # # print("r2 score:", r2)

# # # # # # # # # # # # # # # # # # # #열= column= 속성 =FEATURE

# # # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # # from tensorflow.keras.models import Sequential
# # # # # # # # # # # # # # # # # # # # from tensorflow.keras.layers import Dense
# # # # # # # # # # # # # # # # # # # # from sklearn.model_selection import train_test_split


# # # # # # # # # # # # # # # # # # # # #1 data

# # # # # # # # # # # # # # # # # # # # x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
# # # # # # # # # # # # # # # # # # # # y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])

# # # # # # # # # # # # # # # # # # # # #2 model
# # # # # # # # # # # # # # # # # # # # x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=3)

# # # # # # # # # # # # # # # # # # # # model  = Sequential()
# # # # # # # # # # # # # # # # # # # # model.add(Dense(10, input_dim = 1))
# # # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # # model.add(Dense(10))
# # # # # # # # # # # # # # # # # # # # model.add(Dense(1))


# # # # # # # # # # # # # # # # # # # # #3 compile and train
# # # # # # # # # # # # # # # # # # # # model.compile(loss = 'mse', optimizer = 'adam')

# # # # # # # # # # # # # # # # # # # # model.fit(x_train, y_train, epochs = 500, batch_size = 1)

# # # # # # # # # # # # # # # # # # # # #4 evaluate and predict

# # # # # # # # # # # # # # # # # # # # loss = model.evaluate(x_test, y_test)

# # # # # # # # # # # # # # # # # # # # result = model.predict([x_test])

# # # # # # # # # # # # # # # # # # # # from sklearn.metrics import mean_squared_error

# # # # # # # # # # # # # # # # # # # # def RMSE(y_test, y_predict):
# # # # # # # # # # # # # # # # # # # #     return np.sqrt(mean_squared_error(y_test, y_predict))

# # # # # # # # # # # # # # # # # # # # rmse = RMSE(y_test, result)
# # # # # # # # # # # # # # # # # # # # print("RMSE:", rmse)

# # # # # # # # # # # # # # # # # # # import sklearn as sk
# # # # # # # # # # # # # # # # # # # from sklearn.datasets import load_boston
# # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # from tensorflow.keras.models import Sequential
# # # # # # # # # # # # # # # # # # # from tensorflow.keras.layers import Dense

# # # # # # # # # # # # # # # # # # # from sklearn.model_selection import train_test_split

# # # # # # # # # # # # # # # # # # # dataset = load_boston()

# # # # # # # # # # # # # # # # # # # x = dataset.data
# # # # # # # # # # # # # # # # # # # y = dataset.target


# # # # # # # # # # # # # # # # # # # x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=3)



# # # # # # # # # # # # # # # # # # # #model

# # # # # # # # # # # # # # # # # # # model = Sequential()

# # # # # # # # # # # # # # # # # # # model.add(Dense(50, input_dim = 13))
# # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # # # # model.add(Dense(1))


# # # # # # # # # # # # # # # # # # # model.compile(loss = 'mse', optimizer = 'adam')
# # # # # # # # # # # # # # # # # # # model.fit(x_train, y_train, epochs = 100, batch_size = 1)

# # # # # # # # # # # # # # # # # # # result = model.predict(x_test)

# # # # # # # # # # # # # # # # # # # from sklearn.metrics import r2_score

# # # # # # # # # # # # # # # # # # # r2 = r2_score(y_test, result)
# # # # # # # # # # # # # # # # # # # print("r2 score:", r2)


# # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # import pandas as pd

# # # # # # # # # # # # # # # # # # from tensorflow.keras.models import Sequential
# # # # # # # # # # # # # # # # # # from tensorflow.python.keras.layers import Dense

# # # # # # # # # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # # # # # # # # from sklearn.metrics import r2_score, mean_squared_error
# # # # # # # # # # # # # # # # # # # path = './_data/dacon/따릉이/'
# # # # # # # # # # # # # # # # # # path = './_data/dacon/따릉이/'

# # # # # # # # # # # # # # # # # # train_csv = pd.read_csv(path+'train.csv', index_col=0)
# # # # # # # # # # # # # # # # # # print(train_csv)

# # # # # # # # # # # # # # # # # # test_csv = pd.read_csv(path+ 'test.csv', index_col=0)
# # # # # # # # # # # # # # # # # # print(test_csv)

# # # # # # # # # # # # # # # # # # test_csv = pd.read_csv(path+'test.csv', index_col=0)



# # # # # # # # # # # # # # # # # # submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)
# # # # # # # # # # # # # # # # # # print(submission_csv)


# # # # # # # # # # # # # # # # # # print(train_csv.columns)
# # # # # # # # # # # # # # # # # # print(train_csv.info())
# # # # # # # # # # # # # # # # # # print(train_csv.describe())



# # # # # # # # # # # # # # # # # from tensorflow.keras.models import Sequential
# # # # # # # # # # # # # # # # # from tensorflow.keras.layers import Dense
# # # # # # # # # # # # # # # # # import numpy as np

# # # # # # # # # # # # # # # # # x_train = np.array([1,2,3,4,5,6,7])
# # # # # # # # # # # # # # # # # y_train = np.array([1,2,3,4,5,6,7])

# # # # # # # # # # # # # # # # # x_test = np.array([8,9,10])
# # # # # # # # # # # # # # # # # y_test = np.array([8,9,10])

# # # # # # # # # # # # # # # # # model = Sequential()
# # # # # # # # # # # # # # # # # model.add(Dense(100, input_dim = 1))
# # # # # # # # # # # # # # # # # model.add(Dense(200))
# # # # # # # # # # # # # # # # # model.add(Dense(200))
# # # # # # # # # # # # # # # # # model.add(Dense(200))
# # # # # # # # # # # # # # # # # model.add(Dense(1))

# # # # # # # # # # # # # # # # # model.compile(loss = 'mse', optimizer = 'adam')
# # # # # # # # # # # # # # # # # model.fit(x_train, y_train, epochs = 100, batch_size = 1, verbose = 0)

# # # # # # # # # # # # # # # # # #0 = 침묵, 빨리 넘기기
# # # # # # # # # # # # # # # # # #1 = default 
# # # # # # # # # # # # # # # # # #2 = progress bar delete, 간결해짐
# # # # # # # # # # # # # # # # # #3 = epochs 만 나옴. epoch 만 확인하고 싶으면 0,1,2, 이외의 숫자 입력

# # # # # # # # # # # # # # # # # loss = model.evaluate(x_test, y_test)
# # # # # # # # # # # # # # # # # result = model.predict(np.array([11]))

# # # # # # # # # # # # # # # # # print("loss:", loss)
# # # # # # # # # # # # # # # # # print("[11]의 예측값:", result)


# # # # # # # # # # # # # # # # from tensorflow.keras.models import Sequential
# # # # # # # # # # # # # # # # from tensorflow.keras.layers import Dense
# # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # import time

# # # # # # # # # # # # # # # # x_train = np.array(range(100))
# # # # # # # # # # # # # # # # y_train = np.array(range(100))

# # # # # # # # # # # # # # # # x_test = np.array([8,9,10])
# # # # # # # # # # # # # # # # y_test = np.array([8,9,10])


# # # # # # # # # # # # # # # # model = Sequential()
# # # # # # # # # # # # # # # # model.add(Dense(100, input_dim =1))
# # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # model.add(Dense(100))
# # # # # # # # # # # # # # # # model.add(Dense(1))

# # # # # # # # # # # # # # # # model.compile(loss = 'mse', optimizer = 'adam')
# # # # # # # # # # # # # # # # start_time = time.time() #현재 시간을 반환, 시작시간
# # # # # # # # # # # # # # # # timestamp = 1747971378.293003 
# # # # # # # # # # # # # # # # print(time.ctime(timestamp))
# # # # # # # # # # # # # # # # print(start_time)   #1747971378.293003 

# # # # # # # # # # # # # # # # model.fit(x_train, y_train, epochs =500, batch_size = 2, verbose = 0)


# # # # # # # # # # # # # # # # #0 침묵, 빨리 넘기기
# # # # # # # # # # # # # # # # #1 default
# # # # # # # # # # # # # # # # #2 프로그래스바 삭제, 간결해짐
# # # # # # # # # # # # # # # # #3 에포만 나옴, epoch 만 확인하고 싶으면 0,1,2 이외의 숫자 입력

# # # # # # # # # # # # # # # # end_time = time.time()
# # # # # # # # # # # # # # # # print('걸린시간:', end_time - star)

# # # # # # # # # # # # # # # import pandas as pd

# # # # # # # # # # # # # # # # 파일이 있는 경로
# # # # # # # # # # # # # # # path = 'Study25/_data/kaggle/santander/'


# # # # # # # # # # # # # # # # 1. 불필요한 인덱스가 포함된 CSV 파일 읽기
# # # # # # # # # # # # # # # # 파일 이름은 실제 생성된 파일명으로 변경해주세요.
# # # # # # # # # # # # # # # df = pd.read_csv(path + 'submission_0608_final.csv')

# # # # # # # # # # # # # # # # 2. 필요한 'ID_code'와 'target' 열만 선택
# # # # # # # # # # # # # # # # 파일 형식에 따라 첫 번째 열 이름이 'Unnamed: 0'일 수 있습니다.
# # # # # # # # # # # # # # # # 필요한 두 개의 열만 명시적으로 선택하여 새로운 데이터프레임을 만듭니다.
# # # # # # # # # # # # # # # df_fixed = df[['ID_code', 'target']]

# # # # # # # # # # # # # # # # 3. 인덱스 없이 새로운 파일로 저장
# # # # # # # # # # # # # # # df_fixed.to_csv(path + 'submission_0608_tf_final_fixed.csv', index=False)

# # # # # # # # # # # # # # # print("파일 수정 완료! 'submission_0608_tf_final_fixed.csv' 파일을 확인하세요.")


# # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # import pandas as pd

# # # # # # # # # # # # # # from keras.models import Sequential, Model
# # # # # # # # # # # # # # from keras.layers import Dense, BatchNormalization, Dropout, Input
# # # # # # # # # # # # # # from keras.callbacks import EarlyStopping
# # # # # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # # # # from sklearn.metrics import r2_score, mean_squared_error
# # # # # # # # # # # # # # from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


# # # # # # # # # # # # # # path = 'Study25/_data/kaggle/bank/'

# # # # # # # # # # # # # # train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# # # # # # # # # # # # # # test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# # # # # # # # # # # # # # submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col =0)

# # # # # # # # # # # # # # le_ge0 = LabelEncoder()
# # # # # # # # # # # # # # le_gender = LabelEncoder()

# # # # # # # # # # # # # # train_csv['Geography'] = le_ge0.fit_transform(train_csv['Geography'])
# # # # # # # # # # # # # # train_csv['Gender'] = le_gender.fit_transform(train_csv['Gender'])


# # # # # # # # # # # # # # test_csv['Geography'] = le_ge0.transform(test_csv['Geography'])
# # # # # # # # # # # # # # test_csv['Gender'] = le_gender.transform(test_csv['Gender'])

# # # # # # # # # # # # # # train_csv = train_csv.drop(['CustomerId', 'Surname'], axis = 1)
# # # # # # # # # # # # # # test_csv = test_csv.drop(['CustomerId', 'Surname'], axis=1)

# # # # # # # # # # # # # # # 4. Separate features and target
# # # # # # # # # # # # # # x = train_csv.drop(['Exited'], axis=1)
# # # # # # # # # # # # # # y = train_csv['Exited']


# # # # # # # # # # # # # # x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=33)

# # # # # # # # # # # # # # scaler = StandardScaler()
# # # # # # # # # # # # # # x_train = scaler.fit_transform(x_train)
# # # # # # # # # # # # # # x_test = scaler.transform(x_test)
# # # # # # # # # # # # # # test_csv = scaler.transform(test_csv)

# # # # # # # # # # # # # # model = Sequential([
# # # # # # # # # # # # # #     Dense(128, input_dim = 8, activation='relu'),
# # # # # # # # # # # # # #     Dropout(0.3),
    
# # # # # # # # # # # # # #     Dense(64, activation='relu'),
    
# # # # # # # # # # # # # #     Dense(1)
# # # # # # # # # # # # # # ])

# # # # # # # # # # # # # # model.compile(loss = 'mse', optimizer='adam', metrics = ['acc'])

# # # # # # # # # # # # # # es = EarlyStopping(
# # # # # # # # # # # # # #     monitor= 'val_loss',
# # # # # # # # # # # # # #     mode='min',
# # # # # # # # # # # # # #     patience= 20,
# # # # # # # # # # # # # #     restore_best_weights=True
# # # # # # # # # # # # # # )

# # # # # # # # # # # # # # hist = model.fit(x_train, y_train, epochs = 500, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es])


# # # # # # # # # # # # # # from keras.models import Sequential, Model
# # # # # # # # # # # # # # from keras.layers import Dense, Input, Conv2D, Flatten

# # # # # # # # # # # # # # model = Sequential()
# # # # # # # # # # # # # # model.add(Conv2D(filters = 10, kernel_size=(2,2), input_shape = (5,5,1)))

# # # # # # # # # # # # # # model.add(Conv2D(5, (2,2)))
# # # # # # # # # # # # # # model.add(Flatten())

# # # # # # # # # # # # # # model.add(Dense(units = 10))
# # # # # # # # # # # # # # model.add(Dense(3))


# # # # # # # # # # # # # # model = Sequential()
# # # # # # # # # # # # # # model.add(Conv2D(filters =10, kernel_size=(2,2), input_shape = (5,5,1)))
# # # # # # # # # # # # # # (4,4,10)

# # # # # # # # # # # # # # model.add(Conv2D(filters= 5, kernel_size=(2,2)))
# # # # # # # # # # # # # # (3,3,5)


# # # # # # # # # # # # # # 함수 표현 
# # # # # # # # # # # # # # model1 = Model(inputs = input, outputs = output)

# # # # # # # # # # # # # # model.compile(
# # # # # # # # # # # # # #     loss = 'binary_crossentropy',
# # # # # # # # # # # # # #     loss = 'categorical_crossentropy'
# # # # # # # # # # # # # # )

# # # # # # # # # # # # # from keras.models import Sequential, Model
# # # # # # # # # # # # # from keras.layers import Conv2D, Dense, Dropout, Input, Flatten

# # # # # # # # # # # # # model = Sequential()
# # # # # # # # # # # # # model.add(Conv2D(filters = 11, kernel_size=(2,2), input_shape = (5,5,1))) #(N,5,5,1)

# # # # # # # # # # # # # model.add(Conv2D(filters=7, kernel_size=(2,2))) 

# # # # # # # # # # # # # model.add(Flatten())

# # # # # # # # # # # # # model.add(Dense(units=10))


# # # # # # # # # # # # # model.add(Dense(3))
# # # # # # # # # # # # # model.summary()


# # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # import pandas as pd
# # # # # # # # # # # # from keras.models import Sequential, Model, load_model
# # # # # # # # # # # # from keras.layers import Dense, Input, Conv2D, Dropout, BatchNormalization
# # # # # # # # # # # # from keras.callbacks import EarlyStopping, ModelCheckpoint
# # # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # # from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
# # # # # # # # # # # # from sklearn.datasets import fetch_california_housing

# # # # # # # # # # # # path = 'st/d/t/d//'

# # # # # # # # # # # # train_csv = pd.read_csv(path + 'train.csv', index_col=0)

# # # # # # # # # # # import numpy as np
# # # # # # # # # # # import pandas as pd

# # # # # # # # # # # from keras.datasets import mnist
# # # # # # # # # # # from keras.models import Sequential, Model
# # # # # # # # # # # from keras.layers import Dropout, Input, BatchNormalization, Dense, Conv2D, Flatten

# # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
# # # # # # # # # # # import time

# # # # # # # # # # # from sklearn.metrics import accuracy_score

# # # # # # # # # # # (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # # # # # # # # # # print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# # # # # # # # # # # print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# # # # # # # # # # # #1 MinMaxScaler

# # # # # # # # # # # x_train = x_train.reshape(60000, 28*28)
# # # # # # # # # # # x_test = x_test.reshape(10000, 28*28)

# # # # # # # # # # # scaler = MinMaxScaler()
# # # # # # # # # # # x_train = scaler.fit_transform(x_train)
# # # # # # # # # # # x_test = scaler.transform(x_test)

# # # # # # # # # # # x_train.reshape(60000,28,28,1)
# # # # # # # # # # # x_test.reshape(10000,28,28,1)


# # # # # # # # # # # #2 scaling 
# # # # # # # # # # # x_train = x_train/255
# # # # # # # # # # # x_test = x_test/255

# # # # # # # # # # # #3 scaling
# # # # # # # # # # # x_train = (x_train-127.5)/127.5
# # # # # # # # # # # x_test = (x_test-127.5)/127.5


# # # # # # # # # # # y_train = pd.get_dummies(y_train)
# # # # # # # # # # # y_test = pd.get_dummies(y_test)


# # # # # # # # # # # from keras.models import Sequential
# # # # # # # # # # # from keras.models import Model
# # # # # # # # # # # from keras.models import load_model
# # # # # # # # # # # from keras.layers import Dense
# # # # # # # # # # # from keras.layers import Dropout
# # # # # # # # # # # from keras.layers import BatchNormalization
# # # # # # # # # # # from keras.layers import Input
# # # # # # # # # # # from keras.layers import Conv2D
# # # # # # # # # # # from keras.layers import Flatten

# # # # # # # # # # # from keras.callbacks import EarlyStopping
# # # # # # # # # # # from keras.callbacks import ModelCheckpoint
# # # # # # # # # # # from keras.callbacks import ReduceLROnPlateau
# # # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # # from sklearn.metrics import mean_squared_error
# # # # # # # # # # # from sklearn.metrics import r2_score
# # # # # # # # # # # from sklearn.metrics import accuracy_score
# # # # # # # # # # # from sklearn.metrics import f1_score

# # # # # # # # # # # from sklearn.preprocessing import LabelEncoder
# # # # # # # # # # # from sklearn.preprocessing import OrdinalEncoder
# # # # # # # # # # # from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
# # # # # # # # # # # import numpy as np
# # # # # # # # # # # import pandas as pd
# # # # # # # # # # # import time

# # # # # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # # # # import datetime

# # # # # # # # # # # from sklearn.datasets import fetch_california_housing # 선형 회귀
# # # # # # # # # # # from sklearn.datasets import load_breast_cancer       # 이진 분류
# # # # # # # # # # # from sklearn.datasets import load_iris                # 다중 분류
# # # # # # # # # # # from sklearn.datasets import fetch_covtype
# # # # # # # # # # # from sklearn.datasets import fetch_california_housing
# # # # # # # # # # # from sklearn.datasets import load_boston              # 윤리적 문제로 deprecated (boston 가상환경에서만 사용할 수 있음)
# # # # # # # # # # # from keras.datasets import mnist


# # # # # # # # # # # dataset = load_iris()
# # # # # # # # # # # print(dataset)
# # # # # # # # # # # print(dataset.DESCR)


# # # # # # # # # # import numpy as np
# # # # # # # # # # import pandas as pd

# # # # # # # # # # from keras.datasets import mnist
# # # # # # # # # # from keras.models import Sequential, Model
# # # # # # # # # # from keras.layers import Dense, Input, Conv2D, Dropout, Flatten, BatchNormalization

# # # # # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # # # # from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

# # # # # # # # # # import time
# # # # # # # # # # from sklearn.metrics import accuracy_score

# # # # # # # # # # #1 data
# # # # # # # # # # (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # # # # # # # # # print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# # # # # # # # # # print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

# # # # # # # # # # #1 MinmaxScaling
# # # # # # # # # # x_train = x_train.reshape(60000,28*28)
# # # # # # # # # # x_test = x_test.reshape(10000, 28*28)

# # # # # # # # # # print(x_train.shape, x_test.shape)  #(60000, 784) (10000, 784)

# # # # # # # # # # scaler = MinMaxScaler()
# # # # # # # # # # x_train = scaler.fit_transform(x_train)
# # # # # # # # # # x_test = scaler.transform(x_test)

# # # # # # # # # # x_train = x_train.reshape(60000, 28, 28, 1)
# # # # # # # # # # x_test = x_test.reshape(10000,28,28,1)

# # # # # # # # # # y_train = pd.get_dummies(y_train)
# # # # # # # # # # y_test = pd.get_dummies(y_test)



# # # # # # # # # # #2 model
# # # # # # # # # # model = Sequential([
# # # # # # # # # #     Conv2D(filters=64, kernel_size=(3,3), strides=1, input_shape = (28,28,1), activation='relu'),
# # # # # # # # # #     Dropout(0.2),
# # # # # # # # # #     Conv2D(filters=64, kernel_size=(3,3)),
# # # # # # # # # #     Dropout(0.2),
# # # # # # # # # #     BatchNormalization(),
# # # # # # # # # #     Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
# # # # # # # # # #     Flatten(),
# # # # # # # # # #     Dense(units=16, activation='relu'),
# # # # # # # # # #     Dropout(0.2),
# # # # # # # # # #     Dense(units=16, input_shape = (16,)),
# # # # # # # # # #     Dense(units=10, activation='softmax')
    
# # # # # # # # # # ])

# # # # # # # # # # model.summary()

# # # # # # # # # # model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])


# # # # # # # # # # from keras.callbacks import EarlyStopping, ModelCheckpoint

# # # # # # # # # # es = EarlyStopping(
# # # # # # # # # #     monitor='val_loss',
# # # # # # # # # #     mode = 'auto',
# # # # # # # # # #     patience=20,
# # # # # # # # # #     restore_best_weights=True
# # # # # # # # # # )

# # # # # # # # # # ### save file name###
# # # # # # # # # # import datetime
# # # # # # # # # # date = datetime.datetime.now()
# # # # # # # # # # date = date.strftime("%m%h_%H%M")
# # # # # # # # # # path = 'Study25/_save/keras36_cnn5/'
# # # # # # # # # # filename = '{epoch:04d}-{val_loss:.4f}.h5'
# # # # # # # # # # filepath = "".join([path, 'k36_', date, '_', filename])

# # # # # # # # # # mcp = ModelCheckpoint(
# # # # # # # # # #     monitor='val_loss',
# # # # # # # # # #     mode = 'auto',
# # # # # # # # # #     save_best_only= True,
# # # # # # # # # #     filepath=filepath
# # # # # # # # # # )

# # # # # # # # # # start = time.time()
# # # # # # # # # # hist = model.fit(x_train,y_train, epochs=500, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es, mcp])

# # # # # # # # # # end = time.time()

# # # # # # # # # # loss, acc = model.evaluate(x_test, y_test, verbose = 1)


# # # # # # # # # # print('loss:', loss)
# # # # # # # # # # print('acc:', acc)

# # # # # # # # # # print('걸린시간:', end-start)

# # # # # # # # # # y_pred = model.predict(x_test)
# # # # # # # # # # y_pred = np.argmax(y_pred, axis=1)
# # # # # # # # # # y_test = np.argmax(y_test, axis=1)
# # # # # # # # # # acc_score = accuracy_score(y_test, y_pred)


# # # # # # # # # import pandas as pd
# # # # # # # # # import numpy as np
# # # # # # # # # import os
# # # # # # # # # from datetime import datetime
# # # # # # # # # from imblearn.over_sampling import SMOTE

# # # # # # # # # from keras.models import Sequential
# # # # # # # # # from keras.layers import Dense, Dropout, BatchNormalization
# # # # # # # # # from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# # # # # # # # # from keras.regularizers import l2
# # # # # # # # # from keras.metrics import AUC, Precision, Recall

# # # # # # # # # from sklearn.model_selection import StratifiedShuffleSplit
# # # # # # # # # from sklearn.preprocessing import RobustScaler
# # # # # # # # # from sklearn.metrics import f1_score
# # # # # # # # # from sklearn.impute import SimpleImputer
# # # # # # # # # from xgboost import XGBClassifier

# # # # # # # # # # === Random Seed and Path ===
# # # # # # # # # np.random.seed(333)

# # # # # # # # # # === Set Time-based Save Path ===
# # # # # # # # # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# # # # # # # # # base_path = f'/Users/jaewoo/Desktop/IBM x RedHat/Study25/_save/dacon_cancer/{timestamp}'
# # # # # # # # # os.makedirs(base_path, exist_ok=True)

# # # # # # # # # # === Load Data ===
# # # # # # # # # data_path = '/Users/jaewoo/Desktop/IBM x RedHat/Study25/_data/dacon/cancer/'
# # # # # # # # # train = pd.read_csv(data_path + 'train.csv', index_col=0)
# # # # # # # # # test = pd.read_csv(data_path + 'test.csv', index_col=0)
# # # # # # # # # submission = pd.read_csv(data_path + 'sample_submission.csv', index_col=0)

# # # # # # # # # # === Separate Label ===
# # # # # # # # # x = train.drop(columns=['Cancer'])
# # # # # # # # # y = train['Cancer']

# # # # # # # # # # === Combine train+test for same preprocessing ===
# # # # # # # # # x_all = pd.concat([x, test])
# # # # # # # # # x_all.reset_index(drop=True, inplace=True)

# # # # # # # # # # === Imputation for numerical features ===
# # # # # # # # # num_cols = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']
# # # # # # # # # x_all[num_cols] = SimpleImputer(strategy='mean').fit_transform(x_all[num_cols])

# # # # # # # # # # === One-Hot Encode Categorical Features ===
# # # # # # # # # cat_cols = x_all.select_dtypes(include='object').columns
# # # # # # # # # x_all = pd.get_dummies(x_all, columns=cat_cols)

# # # # # # # # # # === Align train/test again ===
# # # # # # # # # x = x_all.iloc[:len(train)]
# # # # # # # # # x_test = x_all.iloc[len(train):]

# # # # # # # # # # === Feature Scaling ===
# # # # # # # # # scaler = RobustScaler()
# # # # # # # # # x_scaled = scaler.fit_transform(x)
# # # # # # # # # x_test_scaled = scaler.transform(x_test)

# # # # # # # # # # === Feature Selection (XGBoost) ===
# # # # # # # # # xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
# # # # # # # # # xgb.fit(x_scaled, y)
# # # # # # # # # importances = xgb.feature_importances_
# # # # # # # # # threshold = np.percentile(importances, 25)
# # # # # # # # # selected_idx = np.where(importances > threshold)[0]
# # # # # # # # # x_selected = x_scaled[:, selected_idx]
# # # # # # # # # x_test_selected = x_test_scaled[:, selected_idx]

# # # # # # # # # # === Stratified Split ===
# # # # # # # # # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# # # # # # # # # for train_idx, val_idx in sss.split(x_selected, y):
# # # # # # # # #     x_train, x_val = x_selected[train_idx], x_selected[val_idx]
# # # # # # # # #     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# # # # # # # # # # === SMOTE ===
# # # # # # # # # x_train, y_train = SMOTE(random_state=42).fit_resample(x_train, y_train)

# # # # # # # # # # === Build DNN ===
# # # # # # # # # model = Sequential([
# # # # # # # # #     Dense(128, input_dim=x_selected.shape[1], activation='relu'),
# # # # # # # # #     BatchNormalization(),
# # # # # # # # #     Dropout(0.3),
# # # # # # # # #     Dense(64, activation='relu'),
# # # # # # # # #     BatchNormalization(),
# # # # # # # # #     Dropout(0.2),
# # # # # # # # #     Dense(32, activation='relu'),
# # # # # # # # #     BatchNormalization(),
# # # # # # # # #     Dropout(0.1),
# # # # # # # # #     Dense(1, activation='sigmoid')
# # # # # # # # # ])

# # # # # # # # # # === Compile ===
# # # # # # # # # model.compile(
# # # # # # # # #     loss='binary_crossentropy',
# # # # # # # # #     optimizer='adam',
# # # # # # # # #     metrics=['accuracy', AUC(name='auc'), Precision(), Recall()]
# # # # # # # # # )

# # # # # # # # # # === Callbacks ===
# # # # # # # # # model_path = os.path.join(base_path, 'best_model.h5')
# # # # # # # # # mcp = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)
# # # # # # # # # es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
# # # # # # # # # lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

# # # # # # # # # # === Train ===
# # # # # # # # # model.fit(
# # # # # # # # #     x_train, y_train,
# # # # # # # # #     validation_data=(x_val, y_val),
# # # # # # # # #     epochs=200,
# # # # # # # # #     batch_size=32,
# # # # # # # # #     callbacks=[mcp, es, lr],
# # # # # # # # #     verbose=1
# # # # # # # # # )

# # # # # # # # # # === Evaluation ===
# # # # # # # # # loss, acc, auc, prec, rec = model.evaluate(x_val, y_val, verbose=0)
# # # # # # # # # y_val_pred = model.predict(x_val).ravel()

# # # # # # # # # thresholds = np.arange(0.05, 0.95, 0.01)
# # # # # # # # # f1s = [f1_score(y_val, (y_val_pred > t).astype(int)) for t in thresholds]
# # # # # # # # # best_idx = np.argmax(f1s)
# # # # # # # # # best_threshold = thresholds[best_idx]
# # # # # # # # # best_f1 = f1s[best_idx]

# # # # # # # # # # === 결과 출력 ===
# # # # # # # # # print(f'✅ Loss: {loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}')
# # # # # # # # # print(f'✅ Precision: {prec:.4f} | Recall: {rec:.4f}')
# # # # # # # # # print(f'✅ Best F1: {best_f1:.4f} @ Threshold: {best_threshold:.2f}')

# # # # # # # # # # === Predict on Test Set ===
# # # # # # # # # y_test_pred = model.predict(x_test_selected).ravel()
# # # # # # # # # submission['Cancer'] = (y_test_pred > best_threshold).astype(int)

# # # # # # # # # # === Save ===
# # # # # # # # # submission_path = os.path.join(base_path, f'submission_f1_{best_f1:.4f}.csv')
# # # # # # # # # submission.to_csv(submission_path)
# # # # # # # # # print(f'✅ Submission saved to: {submission_path}')


# # # # # # # # import pandas as pd
# # # # # # # # import numpy as np
# # # # # # # # import matplotlib.pyplot as plt
# # # # # # # # from xgboost import XGBClassifier
# # # # # # # # from sklearn.preprocessing import LabelEncoder
# # # # # # # # from sklearn.model_selection import train_test_split

# # # # # # # # # 데이터 불러오기
# # # # # # # # df = pd.read_csv("/Users/jaewoo/Desktop/IBM x RedHat/Study25/_data/dacon/cancer/train.csv")  # 또는 네가 가진 DataFrame

# # # # # # # # # 타겟과 피처 분리
# # # # # # # # X = df.drop(columns=["ID", "Cancer"])  # ID 제거
# # # # # # # # y = df["Cancer"]

# # # # # # # # # 범주형 변수 인코딩
# # # # # # # # for col in X.select_dtypes(include="object").columns:
# # # # # # # #     le = LabelEncoder()
# # # # # # # #     X[col] = le.fit_transform(X[col])

# # # # # # # # # 학습/검증 나누기
# # # # # # # # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # # # # # # # # XGBoost 모델 학습
# # # # # # # # model = XGBClassifier(random_state=42)
# # # # # # # # model.fit(X_train, y_train)

# # # # # # # # # 중요도 추출
# # # # # # # # importances = model.feature_importances_
# # # # # # # # feature_names = X.columns

# # # # # # # # # 중요도 시각화
# # # # # # # # importance_df = pd.DataFrame({
# # # # # # # #     "Feature": feature_names,
# # # # # # # #     "Importance": importances
# # # # # # # # }).sort_values(by="Importance", ascending=False)

# # # # # # # # plt.figure(figsize=(10, 6))
# # # # # # # # plt.barh(importance_df["Feature"], importance_df["Importance"])
# # # # # # # # plt.xlabel("Importance")
# # # # # # # # plt.title("Feature Importance (XGBoost)")
# # # # # # # # plt.gca().invert_yaxis()
# # # # # # # # plt.tight_layout()
# # # # # # # # plt.show()


# # # # # # # import numpy as np
# # # # # # # from keras.preprocessing.text import Tokenizer
# # # # # # # from keras.models import Sequential
# # # # # # # from keras.layers import Dense, LSTM

# # # # # # # import time
# # # # # # # from sklearn.preprocessing import MinMaxScaler, StandardScaler
# # # # # # # from sklearn.model_selection import train_test_split
# # # # # # # from keras.utils import to_categorical
# # # # # # # import pandas as pd

# # # # # # # #1. 데이터
# # # # # # # docs = [
# # # # # # #     '너무 재미있다','참 최고에요', '참 잘만든 영화예요',
# # # # # # #     '추천하고 싶은 영화입니다.','한 번 더 보고 싶어요.','글쎄',
# # # # # # #     '별로에요','생각보다 지루해요','연기가 어색해요',
# # # # # # #     '재미없어요','너무 재미없다.','참 재밌네요.',
# # # # # # #     '석준이 바보','준희 잘생겼다','이삭이 또 구라친다',
# # # # # # # ]

# # # # # # # labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

# # # # # # # predict =['이삭이 참 잘생겼다']

# # # # # # # token = Tokenizer()
# # # # # # # token.fit_on_texts(docs)
# # # # # # # print(token.word_index)
# # # # # # # print(token.word_counts)

# # # # # # # x = token.texts_to_sequences(docs)
# # # # # # # x_text = token.texts_to_sequences(predict)

# # # # # # # ###padding

# # # # # # # from keras.preprocessing.sequence import pad_sequences
# # # # # # # padding_x= pad_sequences(x,
# # # # # # #                          padding='pre',
# # # # # # #                          maxlen=5,
# # # # # # #                          truncating='pre')
# # # # # # # padding_pre = pad_sequences(x_text,
# # # # # # #                             padding='pre',
# # # # # # #                             maxlen=5,
# # # # # # #                             truncating='pre')

# # # # # # # from sklearn.preprocessing import OneHotEncoder
# # # # # # # encoder = OneHotEncoder(sparse_output=False)

# # # # # # # # print(padding_x.shape)

# # # # # # # padding_x = padding_x.reshape(-1,1)
# # # # # # # padding_x = encoder.fit_transform(padding_x)
# # # # # # # padding_x = padding_x[:, 1:]
# # # # # # # padding_x = padding_x.reshape(15,5,30)


# # # # # # # print(padding_x.shape)
# # # # # # # # padding_x = padding_x.reshape(15,5,30)

# # # # # # # padding_pre = padding_pre.reshape(-1,1)
# # # # # # # padding_pre = encoder.transform(padding_pre)
# # # # # # # padding_pre = padding_pre[:, 1:]
# # # # # # # padding_pre = padding_pre.reshape(1,5,30)

# # # # # # # x_train, x_test, y_train, y_test = train_test_split(padding_x, labels, train_size=0.9, random_state=42)

# # # # # # # model = Sequential()
# # # # # # # model.add(LSTM(40, input_shape = (5,30), activation='relu'))
# # # # # # # model.add(Dense(20, activation='relu'))
# # # # # # # model.add(Dense(1, activation='sigmoid'))

# # # # # # # #3 compile and train
# # # # # # # model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
# # # # # # # from keras.callbacks import EarlyStopping, ModelCheckpoint
# # # # # # # es = EarlyStopping(
# # # # # # #     monitor='val_loss',
# # # # # # #     mode = 'min',
# # # # # # #     patience=20,
# # # # # # #     verbose=1,
# # # # # # #     restore_best_weights=True
# # # # # # # )

# # # # # # # hist = model.fit(x_train,y_train, epochs = 100, batch_size=32, callbacks=[es])

# # # # # # # loss, acc = model.evaluate(x_test, y_test, verbose=1)
# # # # # # # result = model.predict(x_test)
# # # # # # # result = (result>0.5).astype(int)

# # # # # # # print('loss:', loss)
# # # # # # # print('acc:', acc)

# # # # # # # print('result:', result)


# # # # # # # prediction = model.predict(padding_pre)
# # # # # # # prediction = (prediction>0.5).astype(int)

# # # # # # # def call(prediction):
# # # # # # #     if prediction ==0:
# # # # # # #         print('negative')
# # # # # # #     else:
# # # # # # #         print('positive')
# # # # # # # call(prediction)


# # # # # # import numpy as np
# # # # # # from keras.preprocessing.text import Tokenizer
# # # # # # from keras.models import Sequential
# # # # # # from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten
# # # # # # import time

# # # # # # from sklearn.preprocessing import MinMaxScaler, StandardScaler
# # # # # # from sklearn.model_selection import train_test_split
# # # # # # from keras.utils import to_categorical


# # # # # # #1. 데이터
# # # # # # docs = [
# # # # # #     '너무 재미있다','참 최고에요', '참 잘만든 영화예요',
# # # # # #     '추천하고 싶은 영화입니다.','한 번 더 보고 싶어요.','글쎄',
# # # # # #     '별로에요','생각보다 지루해요','연기가 어색해요',
# # # # # #     '재미없어요','너무 재미없다.','참 재밌네요.',
# # # # # #     '석준이 바보','준희 잘생겼다','이삭이 또 구라친다',
# # # # # # ]

# # # # # # labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

# # # # # # predict =['이삭이 참 잘생겼다']


# # # # # # token = Tokenizer()
# # # # # # token.fit_on_texts(docs)
# # # # # # print(token.word_index)

# # # # # # x = token.texts_to_sequences(docs)
# # # # # # x_text = token.texts_to_sequences(predict)

# # # # # # from keras.preprocessing.sequence import pad_sequences
# # # # # # padding_x = pad_sequences(x,
# # # # # #                           padding='pre',
# # # # # #                           maxlen=5,
# # # # # #                           truncating='pre')

# # # # # # padding_pred = pad_sequences(x_text,
# # # # # #                              padding='pre',
# # # # # #                              maxlen=5,
# # # # # #                              truncating='pre')


# # # # # # from sklearn.preprocessing import OneHotEncoder
# # # # # # encoder = OneHotEncoder(sparse_output=False)

# # # # # # padding_x = padding_x.reshape(-1,1)
# # # # # # padding_x = encoder.fit_transform(padding_x)
# # # # # # padding_x = padding_x[:, 1:]
# # # # # # padding_x = padding_x.reshape(15,5,30)

# # # # # # padding_pred = padding_pred.reshape(-1,1)
# # # # # # padding_pred = encoder.transform(padding_pred)
# # # # # # padding_pred = padding_pred[:, 1:]
# # # # # # padding_pred = padding_pred.reshape(1,5,30)

# # # # # # x_train, x_test, y_train, y_test = train_test_split(padding_x, labels, train_size=0.9, random_state=42)

# # # # # # model = Sequential()
# # # # # # model.add(Conv1D(filters=40, input_shape = (5,30), kernel_size=2, activation='relu'))
# # # # # # model.add(Flatten())
# # # # # # model.add(BatchNormalization())
# # # # # # model.add(Dense(20, activation='relu'))
# # # # # # model.add(BatchNormalization())
# # # # # # model.add(Dense(1, activation='sigmoid'))

# # # # # # model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics = ['acc'])

# # # # # # from keras.callbacks import EarlyStopping, ModelCheckpoint
# # # # # # es = EarlyStopping(monitor = 'val_loss', mode=min,
# # # # # #                    restore_best_weights=True,
# # # # # #                    patience=20)

# # # # # # model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[es])

# # # # # # loss, acc = model.evaluate(x_test, y_test, verbose = 1)
# # # # # # result = model.predict(x_test)
# # # # # # result = (result>0.5).astype(int)
# # # # # # print('loss :',loss)
# # # # # # print('acc :',acc)
# # # # # # print('result:',result)

# # # # # # predict = model.predict(padding_pred)
# # # # # # predict = (predict>0.5).astype(int)

# # # # # # print(predict)

# # # # # # if predict==0:
# # # # # #     print('negative')
# # # # # # else:
# # # # # #     print('positive')

# # # # # import numpy as np
# # # # # from keras.preprocessing.text import Tokenizer
# # # # # from keras.models import Sequential
# # # # # from keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten
# # # # # import time

# # # # # from sklearn.preprocessing import MinMaxScaler, StandardScaler
# # # # # from sklearn.model_selection import train_test_split
# # # # # from keras.utils import to_categorical

# # # # # #1 data
# # # # # docs = [
# # # # #     '너무 재미있다','참 최고에요', '참 잘만든 영화예요',
# # # # #     '추천하고 싶은 영화입니다.','한 번 더 보고 싶어요.','글쎄',
# # # # #     '별로에요','생각보다 지루해요','연기가 어색해요',
# # # # #     '재미없어요','너무 재미없다.','참 재밌네요.',
# # # # #     '석준이 바보','준희 잘생겼다','이삭이 또 구라친다',
# # # # #     '구라친다','좋아요','슬퍼요','행복해요','나는 오늘 학교에서 너무 재미없다.'
# # # # # ]

# # # # # labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0])

# # # # # predict =['이삭이 참 잘생겼다']

# # # # # token = Tokenizer()
# # # # # token.fit_on_texts(docs)
# # # # # print(token.word_index)
# # # # # print(token.word_counts)


# # # # # # {'너무': 1, '참': 2, '재미없다': 3, '구라친다': 4, '재미있다': 5, '최고에요': 6, '잘만든': 7, '영화예요': 8, '추천하고': 9, '싶은': 10, '영화입니다': 11, '한': 12, '번': 13, '더': 14, '보고': 15, '싶어요': 16, '글쎄': 17, '별로에요': 18, '생각보다': 19, '지루해요': 20, '연기가': 21, '어색해요': 22, '재미없어요': 23, '재밌네요': 24, '석준이': 25, '바보': 26, '준희': 27, '잘생겼다': 28, '이삭이': 29, '또': 30, '좋아요': 31, '슬퍼요': 32, '행복해요': 33, '나는': 34, '오늘': 35, '학교에서': 36}
# # # # # # OrderedDict([('너무', 3), ('재미있다', 1), ('참', 3), ('최고에요', 1), ('잘만든', 1), ('영화예요', 1), ('추천하고', 1), ('싶은', 1), ('영화입니다', 1), ('한', 1), ('번', 1), ('더', 1), ('보고', 1), ('싶어요', 1), ('글쎄', 1), ('별로에요', 1), ('생각보다', 1), ('지루해요', 1), ('연기가', 1), ('어색해요', 1), ('재미없어요', 1), ('재미없다', 2), ('재밌네요', 1), ('석준이', 1), ('바보', 1), ('준희', 1), ('잘생겼다', 1), ('이삭이', 1), ('또', 1), ('구라친다', 2), ('좋아요', 1), ('슬퍼요', 1), ('행복해요', 1), ('나는', 1), ('오늘', 1), ('학교에서', 1)])

# # # # # x = token.texts_to_sequences(docs)
# # # # # x_text = token.texts_to_sequences(predict)

# # # # # #padding
# # # # # from keras.preprocessing.sequence import pad_sequences
# # # # # padding_x = pad_sequences(
# # # # #     x,
# # # # #     padding='pre',
# # # # #     maxlen=5,
# # # # #     truncating='pre',
# # # # # )

# # # # # padding_pred = pad_sequences(
# # # # #     x_text,
# # # # #     padding='pre',
# # # # #     maxlen = 5,
# # # # #     truncating='pre',
# # # # # )

# # # # # from sklearn.preprocessing import OneHotEncoder
# # # # # encoder = OneHotEncoder(sparse_output=False)
# # # # # print(padding_x.shape) #(20,5)
# # # # # print(padding_pred.shape)

# # # # # padding_x = padding_x.reshape(-1,1)
# # # # # padding_x = encoder.fit_transform(padding_x)
# # # # # padding_x = padding_x[:, 1:]
# # # # # padding_x = padding_x.reshape(20,5,36)

# # # # # padding_pred = padding_pred.reshape(-1,1)
# # # # # padding_pred = encoder.transform(padding_pred)
# # # # # padding_pred = padding_pred[:, 1:]
# # # # # padding_pred = padding_pred.reshape(-1,5,36)

# # # # # x_train, x_test, y_train, y_test = train_test_split(padding_x, labels, train_size=0.9, random_state=42)

# # # # # model = Sequential()
# # # # # model.add(Conv1D(40, kernel_size= 2, input_shape =(5,36), activation='relu'))
# # # # # model.add(Flatten())
# # # # # model.add(BatchNormalization())
# # # # # model.add(Dense(20, activation='relu'))
# # # # # model.add(BatchNormalization())
# # # # # model.add(Dense(1, activation='sigmoid'))

# # # # # #3 compile and train
# # # # # model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

# # # # # from keras.callbacks import EarlyStopping, ModelCheckpoint
# # # # # es = EarlyStopping(
# # # # #     monitor = 'val_loss',
# # # # #     mode = 'min',
# # # # #     patience=15,
# # # # #     restore_best_weights=True,
# # # # # )

# # # # # hist = model.fit(
# # # # #     x_train, y_train,
# # # # #     epochs = 100,
# # # # #     batch_size = 8,
# # # # #     callbacks = [es]
# # # # # )

# # # # # loss, acc = model.evaluate(x_test, y_test, verbose = 1)
# # # # # result = model.predict(x_test)

# # # # # result = (result>0.5).astype(int)

# # # # # print('loss :',loss)
# # # # # print('acc :',acc)
# # # # # print('result:',result)

# # # # # predict = model.predict(padding_pred)
# # # # # predict = (predict > 0.5).astype(int)
# # # # # print('fp:', predict)
# # # # # if predict==0:
# # # # #     print('negative')
# # # # # else:
# # # # #     print('positive')


# # # # import numpy as np
# # # # from keras.preprocessing.text import Tokenizer
# # # # from keras.models import Sequential, Model
# # # # from keras.layers import Dense, Dropout, BatchNormalization, Embedding, LSTM
# # # # import time
# # # # from sklearn.preprocessing import MinMaxScaler, StandardScaler
# # # # from sklearn.model_selection import train_test_split
# # # # from keras.utils import to_categorical
# # # # import warnings
# # # # warnings.filterwarnings('ignore')

# # # # #1. 데이터
# # # # docs = [
# # # #     '너무 재미있다','참 최고에요', '참 잘만든 영화예요',
# # # #     '추천하고 싶은 영화입니다.','한 번 더 보고 싶어요.','글쎄',
# # # #     '별로에요','생각보다 지루해요','연기가 어색해요',
# # # #     '재미없어요','너무 재미없다.','참 재밌네요.',
# # # #     '석준이 바보','준희 잘생겼다','이삭이 또 구라친다',
# # # # ]

# # # # labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

# # # # predict =['이삭이 참 잘생겼다']


# # # # token = Tokenizer()
# # # # token.fit_on_texts(docs)

# # # # x = token.texts_to_sequences(docs)
# # # # x_text = token.texts_to_sequences(predict)

# # # # from keras.preprocessing.sequence import pad_sequences

# # # # padding_x = pad_sequences(
# # # #     x,
# # # #     padding='pre',
# # # #     truncating= 'pre',
# # # #     maxlen=5
# # # # )

# # # # padding_pred = pad_sequences(
# # # #     x_text,
# # # #     padding='pre',
# # # #     truncating='pre',
# # # #     maxlen=5
# # # # )

# # # # x_train, x_test, y_train, y_test = train_test_split(padding_x, labels, train_size=0.9, random_state=42)


# # # # model = Sequential()
# # # # #embedding 1
# # # # model.add(Embedding(input_dim= 31, output_dim=100, input_length=5))
# # # #     #input_dim = 단어 사전의 개수(말뭉치의 개수)
# # # #     #output_dim = 다음 레어이로 전달하는 노드의 갯수(조절 가능)
# # # #     #input_length = (N,5) padding에 의해서 늘어난 칼럼의 수, 문장의 시퀀스 갯수
# # # # # model.add(Embedding(input_dim=31, output_dim=100, input_length=5))
# # # # model.add(LSTM(16))
# # # # model.add(Dense(1, activation='sigmoid'))

# # # # model.summary()

# # # # model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['acc'])
# # # # from keras.callbacks import EarlyStopping, ModelCheckpoint
# # # # es = EarlyStopping(
# # # #     monitor = 'val_loss',
# # # #     mode = 'min',
# # # #     patience=15,
# # # #     verbose=1,
# # # #     restore_best_weights=True
# # # # )

# # # # hist = model.fit(x_train, y_train, epochs=200, batch_size=4, callbacks=[es])


# # # # loss, acc = model.evaluate(x_test, y_test, verbose =1)

# # # # result = model.predict(x_test)
# # # # result = (result>0.5).astype(int)

# # # # print('loss :',loss)
# # # # print('acc :',acc)
# # # # print('result:',result)

# # # # predict = model.predict(padding_pred)
# # # # print(predict)


# # # # =====================================================
# # # # ✅ DACON Final Submission Template
# # # # ✅ Author: Jaewoo
# # # # ✅ OS: macOS 14.5 (Apple Silicon)
# # # # ✅ Python: 3.10.13
# # # # ✅ 주요 라이브러리 버전
# # # #   pandas==1.5.3
# # # #   numpy==1.23.5
# # # #   scikit-learn==1.3.0
# # # #   xgboost==1.7.6
# # # #   lightgbm==3.3.5
# # # # =====================================================
# # # # =====================================================
# # # # ✅ DACON Final Submission Template (with Jaewoo preprocessing)
# # # # ✅ Author: Jaewoo
# # # # ✅ OS: macOS 14.5 (Apple Silicon)
# # # # ✅ Python: 3.10.13
# # # # ✅ 주요 라이브러리 버전
# # # #   pandas==1.5.3
# # # #   numpy==1.23.5
# # # #   scikit-learn==1.3.0
# # # #   xgboost==1.7.6
# # # # =====================================================
# # # # 1. 라이브러리 로딩
# # # import pandas as pd
# # # import numpy as np
# # # import random
# # # import os
# # # import datetime
# # # from sklearn.model_selection import StratifiedKFold, train_test_split
# # # from sklearn.metrics import f1_score
# # # from imblearn.over_sampling import SMOTE
# # # from xgboost import XGBClassifier

# # # # 2. Seed 고정
# # # SEED = 190
# # # random.seed(SEED)
# # # np.random.seed(SEED)
# # # os.environ['PYTHONHASHSEED'] = str(SEED)

# # # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

# # # # 3. 데이터 로딩
# # # path = '/Users/jaewoo000/Desktop/IBM:Redhat/Study25/_data/dacon/cancer/'
# # # train_csv = pd.read_csv(path+'train.csv', index_col=0)
# # # test_csv = pd.read_csv(path+'test.csv', index_col=0)
# # # submission = pd.read_csv(path+'sample_submission.csv', index_col=0)

# # # # 4. 전처리
# # # train_csv['is_train'] = 1
# # # test_csv['is_train'] = 0
# # # combined = pd.concat([train_csv, test_csv], axis=0)

# # # # 원핫인코딩
# # # combined = pd.get_dummies(combined, columns=['Gender','Country','Race','Family_Background','Radiation_History',
# # #                                              'Iodine_Deficiency','Smoke','Weight_Risk','Diabetes'],
# # #                           drop_first=True, dtype=int)

# # # # 불필요 칼럼 제거
# # # drop_features = ["T3_Result","T4_Result","TSH_Result","Nodule_Size","Age"]
# # # combined.drop(columns=drop_features, inplace=True)

# # # # 다시 분리
# # # train_csv = combined[combined['is_train']==1].drop(columns='is_train')
# # # test_csv = combined[combined['is_train']==0].drop(columns=['is_train','Cancer'])

# # # x = train_csv.drop(['Cancer'], axis=1)
# # # y = train_csv['Cancer']

# # # # 5. Train/Test Split + SMOTE
# # # x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, stratify=y, random_state=334)

# # # smote = SMOTE(random_state=SEED)
# # # x_train, y_train = smote.fit_resample(x_train, y_train)

# # # # 6. KFold Ensemble with ModelCheckpoint
# # # kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
# # # test_preds = np.zeros(test_csv.shape[0])
# # # val_scores = []

# # # for fold, (train_idx, val_idx) in enumerate(kf.split(x, y)):
# # #     print(f"\n🚀 Fold {fold+1}")

# # #     x_tr, x_va = x.iloc[train_idx], x.iloc[val_idx]
# # #     y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

# # #     # SMOTE 각 fold에 적용
# # #     x_tr, y_tr = smote.fit_resample(x_tr, y_tr)

# # #     model = XGBClassifier(
# # #         n_estimators=1000,
# # #         learning_rate=0.05,
# # #         max_depth=5,
# # #         subsample=0.8,
# # #         colsample_bytree=0.8,
# # #         reg_alpha=0.1,
# # #         reg_lambda=1,
# # #         random_state=SEED,
# # #         eval_metric='logloss',
# # #         use_label_encoder=False
# # #     )

# # #     model.fit(
# # #         x_tr, y_tr,
# # #         eval_set=[(x_va, y_va)],
# # #         early_stopping_rounds=50,
# # #         verbose=50
# # #     )

# # #     # ✅ Save model weights for this fold (ModelCheckpoint)
# # #     model_save_path = os.path.join(path, f"xgb_fold{fold+1}_{timestamp}.json")
# # #     model.save_model(model_save_path)
# # #     print(f"💾 Model weights saved to {model_save_path}")

# # #     # Evaluate threshold for best F1
# # #     val_pred = model.predict_proba(x_va)[:,1]
# # #     best_f1, best_th = 0, 0.5
# # #     for th in np.arange(0.3, 0.7, 0.01):
# # #         f1 = f1_score(y_va, (val_pred > th).astype(int))
# # #         if f1 > best_f1:
# # #             best_f1, best_th = f1, th

# # #     print(f"✅ Fold {fold+1} Best F1: {best_f1:.4f} at threshold {best_th:.2f}")
# # #     val_scores.append(best_f1)

# # #     # Test prediction
# # #     test_preds += (model.predict_proba(test_csv)[:,1] > best_th).astype(int) / kf.n_splits

# # # # 7. Submission
# # # submission['Cancer'] = (test_preds > 0.5).astype(int)
# # # filename = os.path.join(path, f"submission_final_{timestamp}.csv")
# # # submission.to_csv(filename)

# # # print(f"\n🎯 KFold Average F1: {np.mean(val_scores):.4f}")
# # # print(f"✅ Submission saved as {filename}")


# # import numpy as np
# # import pandas as pd
# # import datetime
# # import warnings
# # warnings.filterwarnings('ignore')

# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import OneHotEncoder
# # from keras.preprocessing.text import Tokenizer
# # from keras.preprocessing.sequence import pad_sequences
# # from keras.models import Model
# # from keras.layers import Input, Embedding, LSTM, Bidirectional, Dropout, Dense, Concatenate
# # from keras.callbacks import EarlyStopping



# # timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
# # path = './Study25/_data/kaggle/nlp_disaster/'
# # train_csv = pd.read_csv(path + 'train.csv')
# # test_csv = pd.read_csv(path + 'test.csv')
# # submission_csv = pd.read_csv(path + 'sample_submission.csv')


# # #2 결측치 처리
# # for df in [train_csv, test_csv]:
# #     df['keyword'] = df['keyword'].fillna('unknown')
# #     df['location'] = df['location'].fillna('unknown')
# #     df['text'] = df['text'].astype(str)
    
# # token = Tokenizer(num_words=10000, oov_token="<OOV>")
# # token.fit_on_texts(train_csv['text'])

# import numpy as np
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import r2_score
# import warnings

# warnings.filterwarnings('ignore')

# from sklearn.utils import all_estimators
# import sklearn as sk

# x, y = fetch_california_housing(return_X_y=True)

# x_train, x_test, y_train, y_test= train_test_split(x,y, shuffle=True, random_state=42)

# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)


# allAlgorithms = all_estimators(type_filter='regressor')
# print('모델의 갯수:', len(allAlgorithms))

# max_score = 0
# max_name = []

# for (name, algorithm) in allAlgorithms:
#     try:
#         model = algorithm()
#         model.fit(x_train, y_train)
        
#         result = model.score(x_test, y_test)
        
#         if(result > max_score):
#             max_score = result
#             max_name = name
#         print(name, '의 정답률 : ', result)
#     except:
#         print(name, '은(는) 에러뜬 부분!!')
        
# print('======================================================')
# print('최고모델:', max_name, max_score)
# print('======================================================')


# allAlgorithms = all_estimators(type_filter='regressor')
# print('모델의 갯수', len(allAlgorithms))

# allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
# allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='classifier')


# n_split = 5

# import numpy as np
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import r2_score
# import warnings
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.datasets import load_iris
# from sklearn.model_selection import KFold, train_test_split
# from sklearn.model_selection import KFold, train_test_split, cross_val_score
# from sklearn.model_selection import KFold, StratifiedKFold
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import KFold, StratifiedKFold
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# import pandas as pd

# n_split = 5

# kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

# model = MLPClassifier()

# score = cross_val_score(model, x,y, cv=kfold)



import numpy as np
from sklearn.utils.multiclass import type_of_target

def check_target_type(y):
    target_type = type_of_target(y)
    unique_classes = np.unique(y)

    print(f"🎯 데이터 타입: {target_type}")

    if target_type == 'binary':
        print("✅ 이진 분류 (binary classification)")
        print(f"클래스 목록: {unique_classes}")
    
    elif target_type == 'multiclass':
        print("✅ 다중 분류 (multi-class classification)")
        print(f"클래스 개수: {len(unique_classes)}")
        print(f"클래스 목록: {unique_classes}")
    
    elif target_type == 'multilabel-indicator':
        print("✅ 다중 레이블 분류 (multi-label classification)")
        print(f"데이터 shape: {y.shape} → 각 샘플마다 {y.shape[1]}개의 클래스 가능성")
    
    elif target_type == 'continuous':
        print("📉 회귀 문제 (continuous target)")
    
    else:
        print("⚠️ 인식되지 않는 형태입니다. 확인이 필요합니다.")
        
        
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
x, y = data.data, data.target

check_target_type(y)