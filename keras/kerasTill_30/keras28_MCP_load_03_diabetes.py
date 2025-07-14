#copied from 18-3
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

#1 data
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(442, 10) (442,)

# exit()

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.75, random_state=333)



path_mcp = 'Study25/_save/keras28_mcp/03_diabetes/'
model =load_model(path_mcp + 'keras28_diabetes_save.h5')


model.compile(loss ='mse',optimizer = 'adam')




loss = model.evaluate(x_test, y_test)
result = model.predict(x_test)

r2 = r2_score(y_test, result)
print("r2 result:", r2)
