# # # import tensorflow as tf
# # # print('version of tf: ', tf.__version__)

# # # mnist = tf.keras.datasets.mnist

# # # (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # # x_train, x_test = x_train / 255.0, x_test / 255.0

# # # #Flatten is used to reshape the input data into a 1D vector.

# # # #ReLU stands for Rectified Linear Unit and is a very common activation function. 
# # # #ReLU outputs the input directly if it’s positive, and 0 if it’s negative.
# # # #It helps introduce non-linearity into the model, allowing it to learn more complex patterns.
# # # #So if the output of a neuron is prositive, it passes the value as-is. otherwise it outputs zero

# # # #Dropout layer is a refularizationh technique used during training to prevent overfitting. 
# # # #overfitting happens when the model learns too perform very well on the training data but fails to generalize new, unseen data
# # # #dropout rate 0.2 means that 20% of the neurons in layer will be randomly dropped out during each training step.
# # # # ++ During inference, like making predictions, dropout is turned off, and all neurons are used.
 
# # # #Dense(10) -  this is the output layer of the model. number 10 indicated that there are 10 neurons in this layer,
# # # #corresponding to the 10 possible classes for classification. for example in the MNIST dataset, there are 10 possible digits so you need
# # # #10 neurons to represent the 10 possible outputs
# # # #softmax fuction is used in multi-class classification problems. converts the output of the laer into probabiilities that sum to 1
# # # model = tf.keras.models.Sequential(
# # #     [
# # #         tf.keras.layers.Flatten(input_shape=(28,28)),   #reshapes the input image (28x28) into a 1D vector
# # #         tf.keras.layers.Dense(128, activation='relu'),  #Hidden layer with 128 neurons and ReLU activation
# # #         tf.keras.layers.Dropout(0.2),   #Regularization to prevent overfitting by dropping 20% of neurons during training
# # #         tf.keras.layers.Dense(10, activation='softmax') #output layer with 10 neurons (for 10 classes ) and softmax activation
        
# # #     ]
# # # )

# # # #adam is an adaptive optimizer that adjusts the learning rate based on the progress of the model therefore, you dont need to manually tune it
# # # #Loss sparse categorical crossentropy is a loss function that tells the model how well it is performing after each prediction. \
# # # #it calculates the difference between the predicted output and the true output(target) 
# # # #goal during training is to minimize this loss function
# # # #why sparse? 

# # # #Metrics accuracy - metrics are the criteria used to evaluate the performance of the model during training and testing.
# # # #tell you how well the model is doing based on the evaluation metric you choose

# # # # *** optimizer helps model learn by adjusting weights
# # # # loss measures how good or bad the model is at making predictions
# # # # Metrics like accuracy tell us how well the model is performing

# # # model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
# # # predictions = model(x_train[:1]).numpy



# # # import pandas as pd
# # # import numpy as np
# # # data_url = "http://lib.stat.cmu.edu/datasets/boston"
# # # raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# # # data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# # # target = raw_df.values[1::2, 2]

# # # from keras.models import Sequential, load_model
# # # from keras.layers import Dense, BatchNormalization, Dropout
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.metrics import r2_score, mean_squared_error
# # # from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
# # # from keras.callbacks import EarlyStopping, ModelCheckpoint

# # # x = data
# # # y = target

# # # x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=42)

# # # scaler = StandardScaler()
# # # x_train = scaler.fit_transform(x_train)
# # # x_test = scaler.transform(x_test)

# # # model = Sequential()
# # # model.add(Dense(32, input_dim = 13, activation = 'relu'))
# # # model.add(Dropout(0.3))
# # # model.add(BatchNormalization())

# # # model.add(Dense(64, activation = 'relu'))
# # # model.add(Dropout(0.3))
# # # model.add(BatchNormalization())

# # # model.add(Dense(64, activation = 'relu'))
# # # model.add(Dropout(0.3))
# # # model.add(BatchNormalization())

# # # model.add(Dense(1, activation = ''))


# # # path = './_save/keras_mcp/01boston/'
# # # model.save(path + 'keras280fakdlsfjlds.h5')

# # # model.compile(loss = 'mse', optimizer = 'adam', metrics=['acc'])


# # # es = EarlyStopping(
# # #     monitor='val_loss',
# # #     mode = 'auto',
# # #     patience=30,
# # #     restore_best_weights=True
# # # )

# # # import datetime
# # # date = datetime.datetime.now()

# # # date = date .strftime("%m%d_%H%M")

# # # filename = '{epoch:04d}-{val_loss:.4f}.h5'

# # # filepath = "".join([path, 'k28_', date, '_', filename])


# # # mcp = ModelCheckpoint(
# # #     monitor='val_loss',
# # #     mode = 'auto',
# # #     save_best_only= True,
# # #     filepath= filepath
# # # )

# # # hist = model.fit(x_train,y_train, epochs = 10000, batch_size=32, 
# # #                  verbose = 1,
# # #                  validation_split = 0.2,
# # #                  callbacks=[es,mcp])



# # # loss, acc = model.evaluate(x_test, y_test)
# # # print("loss =", loss)
# # # print("acc:", acc)


# # # result = model.predict(x_test)

# # # r2 = r2_score(y_test, result)
# # # rmse = np.sqrt(mean_squared_error(y_test, result))

# # from keras.models import Sequential
# # from keras.models import Model
# # from keras.models import load_model
# # from keras.layers import Dense
# # from keras.layers import Dropout
# # from keras.layers import BatchNormalization
# # from keras.layers import Conv2D
# # from keras.layers import Flatten
# # from keras.layers import Input
# # from keras.callbacks import EarlyStopping
# # from keras.callbacks import ModelCheckpoint

# # from keras.utils import to_categorical

# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import MinMaxScaler
# # from sklearn.preprocessing import OneHotEncoder
# # from sklearn.metrics import accuracy_score

# # import numpy as np
# # import pandas as pd

# # import matplotlib.pyplot as plt
# # import datetime
# # import time

# # from keras.datasets import mnist

# # dataset = mnist()
# # x = dataset.data
# # y = dataset.target

# # path = ''


# # x = x.reshape(60000, 28*28)
# # y = y.reshape(10000, 28*28)

# # # x = x.reshape(x.shape[0],)

# # scaler = MinMaxScaler()
# # x = scaler.fit_transform(x)
# # y = scaler.transform(y)

# # x = x.reshape(60000,28,28,1)
# # y = y.reshape(10000,28,28,1)

# # x = x/255.
# # y = y/255.

# # # x = (x-127.5)/127.5
# # # y = (y-127.5)127.5


# # # onehot

# # y = pd.get_dummies(y)

# # x_train, y_train, x_test, y_test = train_test_split(
# #     x,y, train_size=0.8, shuffle=True, random_state=42, stratify=y
# # )

# # model = Sequential()

# # model.add(Conv2D(filters = 64, kernel_size=(3,3), strides=1, input_shape=(10,10,1)))

# # model.add()


# # import tensorflow as tf
# # print(tf.config.list_physical_devices('GPU'))

# import tensorflow as tf
# import time

# # Simple matrix multiplication benchmark
# device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
# print(f"Using device: {device}")

# a = tf.random.normal([1000, 1000])
# b = tf.random.normal([1000, 1000])

# start = time.time()
# for _ in range(1000):
#     c = tf.matmul(a, b)
# tf.print("Time taken:", time.time() - start, "seconds")


# lkadjsflajfladsjflafjds