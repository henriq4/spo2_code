"""
Carregar os dados
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import pickle

#
# import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.optimizers import RMSprop,Adam

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

from GeradorDeDados import DataGenerator

# =============================================================================
#
# =============================================================================

def create_nn_model(x_shape,y_shape):
    n_timesteps, n_channels, n_outputs = (x_shape[1], x_shape[2], y_shape[1])

    model = Sequential()
    # model.add(Conv1D(32,40,input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=21, activation='relu', input_shape=(n_timesteps,n_channels)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(4)))
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=64, kernel_size=21,activation='relu'))#,input_shape=(size_maxpooling_L1,num_filter_L1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(4)))
    model.add(Dropout(0.1))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    model.summary()
    rmsprop = RMSprop(learning_rate=0.001, rho=0.9)
    model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['mae'])
    return model

data_path = "dados"
train_batch = []

files = os.listdir(data_path)


vols = deque(files)
histories = []

wdw = 256

dnn_results = pd.DataFrame(columns = ['vol','true','pred'])

for i in range(len(vols)):
    test_vol = vols[i]
    test_batch = pd.read_csv(os.path.join(data_path,test_vol), index_col=0)

    model = create_nn_model((1000,256,2), (1000,1))
    # carregar os pesos
    model.load_weights(
        # 'optimized_vRG_70epc_'+test_vol.split('.')[0]+'.h5'
        'optimized_vRG_70epc_'+test_vol.split('.')[0]+'.weights.tflite'
        # 'vRG_70epc_'+test_vol.split('.')[0]+'.h5'
    )
    results = []
    X_test = []
    y_test = []
    for t in range(wdw, len(test_batch), wdw//2):
        batch = test_batch.iloc[t-wdw:t]

        X_test.append(batch[['R','G']].values)
        y_test.append(batch[['S']].mean()[0])

    pred = model.predict(np.array(X_test))[:,0]

    results.append([np.array(y_test), pred])

    dnn_results = pd.concat([dnn_results,
                             pd.DataFrame({"vol":[test_vol]*len(pred),
                                           "true":np.array(y_test)*100,
                                           "pred":pred.astype(np.float64)*100})])

    plt.figure()
    plt.scatter(pred*100,np.array(y_test)*100)
    plt.xlabel('Prediçao')
    plt.ylabel('Referencia')
    plt.title(test_vol)
    plt.show()



rmse = np.sqrt(mse(dnn_results['pred'],dnn_results['true']))
corr = np.corrcoef((dnn_results[['pred','true']].values.T))[0,1]


# dnn_results.to_csv("DNN_hoffman_results.csv")
# dnn_results.to_csv("optimized-DNN_hoffman_results.csv")
dnn_results.to_csv("optimized-tflite-DNN_hoffman_results.csv")

plt.figure()
plt.scatter(dnn_results['pred'],dnn_results['true'])
plt.xlabel('Prediçao')
plt.ylabel('Referencia')
plt.title("RMSE: {:.2f}".format(rmse)+" || "+"correlation: {:.2f}".format(corr))
plt.show()

print("RMSE:",rmse)
print("CORRELACAO:",corr)