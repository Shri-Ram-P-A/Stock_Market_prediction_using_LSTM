# %%
import pandas as pd
import numpy as np
import tensorflow as tf

# %%

data = pd.read_csv("IBM.csv")
data.head()
# %%

df = data['Close']
length = len(df)

# %%

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df=scaler.fit_transform(np.array(df).reshape(-1,1))

# %%

train = df[:int(length*.75)]
test = df[int(length*.75):]

#%%

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
      a = dataset[i:(i+time_step), 0]
      dataX.append(a)
      dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

#%%

x_train, y_train = create_dataset(train, 100)
x_test, y_test = create_dataset(test, 100)

# %%

print(x_train.shape), print(y_train.shape)

# %%

X_train =x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
X_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)

# %%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# %%

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])

# %%

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)

# %%

model.summary()

# %%

tf.__version__

# %%

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

# %%

train_predict

# %%

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

# %%

import numpy 
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
print(math.sqrt(mean_squared_error(y_train,train_predict)))

look_back=100
trainPredictPlot = numpy.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# %%

print(test.shape)

# %%

import numpy as np

# Assuming `test` is a 1D array
x_input = test[-100:].reshape(1, -1)  # Extract last 100 elements and reshape
temp_input = list(x_input[0])  # Convert the array into a list
lst_output = []
n_steps = 100
i = 0

while i < 30:
    if len(temp_input) > 100:
        x_input = np.array(temp_input[-100:])
        print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape((1, n_steps, 1))  
        yhat = model.predict(x_input, verbose=1) 
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())  
        lst_output.extend(yhat.tolist())  
    else:
        x_input = np.array(temp_input).reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)  
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())  
        lst_output.extend(yhat.tolist())  

    i += 1 

print(lst_output)

# %%

day_new=np.arange(1,101)
day_pred=np.arange(101,131)
plt.plot(day_new,scaler.inverse_transform(df[-100:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))

# %%

df3=df.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])

# %%

df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)
