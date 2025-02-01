import os
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np
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
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

from sklearn.metrics import mean_absolute_error as mae

from GeradorDeDados import DataGenerator

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

for i in range(len(vols)):
    vols.rotate(1)
    
    test_vol = vols[0]
    val_batch = [pd.read_csv(os.path.join(data_path,vols[0]), index_col=0)]
    
    for tv in range(1,len(vols)):
        train_batch.append(pd.read_csv(os.path.join(data_path,vols[tv]), index_col=0))    
        print(vols[tv])

    dg_train = DataGenerator(data_list = train_batch, length=256, 
                                    shuffle=True, batch_size=256, n_batches=40)
    
    dg_test = DataGenerator(data_list = val_batch, length=256, 
                                    shuffle=True, batch_size=256, n_batches=1)
    
    model = create_nn_model((1000,256,2), (1000,1))
    
    
    
    # train the model
    EPOCHS= 70
    history = model.fit(x=dg_train, 
                        epochs=EPOCHS, 
                        validation_data=dg_test)
    model.save_weights('vRG_70epc_'+test_vol.split('.')[0]+'.weights.h5')
    # save the historical data
    with open('train_hist_'+test_vol.split('.')[0]+'.p', 'wb') as f:
        pickle.dump({'train_loss':history.history['loss'],'val_loss':history.history['val_loss'],
                     'train_mae':history.history['mae'],'val_mae':history.history['val_mae']},f)
 


plt.figure()
plt.plot(history.history['val_loss'],label = 'val_loss')
plt.plot(history.history['loss'], label = 'loss')
plt.title('Aprendizado da rede')
plt.ylabel('Erro')
plt.xlabel('Época')
plt.legend()
plt.show()

# EVAL

# -*- coding: utf-8 -*-
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
        'vRG_70epc_'+test_vol.split('.')[0]+'.weights.h5'
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


dnn_results.to_csv("DNN_hoffman_results.csv")

plt.figure()
plt.scatter(dnn_results['pred'],dnn_results['true'])
plt.xlabel('Prediçao')
plt.ylabel('Referencia')
plt.title("RMSE: {:.2f}".format(rmse)+" || "+"correlation: {:.2f}".format(corr))
plt.show()

print("RMSE:",rmse)
print("CORRELACAO:",corr)

#     #Save the model with lower training loss
#     model_checkpoint_callback = ModelCheckpoint(
#         filepath='v1_'+test_vol.split('.')[0]+'.hdf5',
#         save_weights_only=True,
#         monitor='loss',
#         mode='min',
#         save_best_only=True)
    
#     model_earlystopping_callback = EarlyStopping(
#         monitor='val_loss',
#         min_delta=0.005,
#         patience=20,
#         verbose=0,
#         mode='min',
#         baseline=None,
#         restore_best_weights=False
#     )
    
#     # train the model
#     EPOCHS= 1000
#     history = model.fit(x=dg_train, 
#                         epochs=EPOCHS, 
#                         validation_data=dg_test,
#                         callbacks = [model_checkpoint_callback,
#                                      model_earlystopping_callback])
    
#     # save the historical data
#     with open('train_hist_'+test_vol.split('.')[0]+'.p', 'wb') as f:
#         pickle.dump({'train_loss':history.history['loss'],'val_loss':history.history['val_loss'],
#                      'train_mae':history.history['mae'],'val_mae':history.history['val_mae']},f)
 


# plt.figure()
# plt.plot(history.history['val_loss'],label = 'val_loss')
# plt.plot(history.history['loss'], label = 'loss')
# plt.title('Aprendizado da rede')
# plt.ylabel('Erro')
# plt.xlabel('Época')
# # plt.legend(['train', 'test'], loc='upper left')
# plt.legend()
# plt.show()


# # myMae = mae(test_Y[:,0], model_Y[:,0])

# # plt.figure()
# # txt = 'Valores preditos pela rede\nErro médio absoluto = {maeValue:.2f} Hz'
# # plt.title(txt.format(maeValue = myMae))
# # plt.plot(test_Y[:,0],label = 'Verdadeiro')
# # plt.plot(model_Y[:,0],label = 'Predito')
# # plt.ylabel('Freq. Resp Hz')
# # plt.xlabel('Amostra')
# # plt.legend()
# # plt.show()

# plt.figure()
# #draw_df.plot()

# Show results

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import matplotlib 
plt.rcParams.update({'font.size': 50})
matplotlib.rc('xtick', labelsize=50) 
matplotlib.rc('ytick', labelsize=50)

def ccc(gold, pred):
    '''
    Function adapted from https://arxiv.org/abs/2003.10724

    Parameters
    ----------
    gold : array
        True measurements.
    pred : array
        Predicted measurements.

    Returns
    -------
    ccc : float (-1 to 1 )
        correlation score.

    '''
    gold_mean  = np.mean(gold)
    pred_mean  = np.mean(pred)
    covariance = np.mean((gold-gold_mean)*(pred-pred_mean))
    gold_var   = np.mean((gold-gold_mean)**2)
    pred_var   = np.mean((pred-pred_mean)**2)
    ccc        = 2 * covariance / (gold_var + pred_var + (gold_mean - pred_mean)**2 + 1e-16)
    return ccc

wdw = 256

dnn_results = pd.read_csv("DNN_hoffman_results.csv")

vols = dnn_results['vol'].unique()




for i in range(len(vols)): 
    print("\nVoluntário",vols[i])
    test_vol =  vols[i]
    batch = dnn_results[dnn_results['vol'] == test_vol].copy()
    pred = batch['pred']
    y_test = batch['true']
    
    rmse = np.sqrt(mse(batch['pred'],batch['true']))
    corr = np.corrcoef((batch[['pred','true']].values.T))[0,1]
    mymae = mae(batch['pred'],batch['true'])

    class_true = np.zeros([len(batch)])
    class_pred = np.zeros([len(batch)])
    class_true[batch['true']<90] = 1
    class_pred[batch['pred']<90] = 1

    
    print("RMSE:", rmse)
    print("CORRELACAO:", corr)
    print("CCC:", ccc(batch['pred'],batch['true']))
    print("MAE:", mymae)
    print("std", np.std(batch['true'] - batch['pred']))
    print("median", np.median(batch['true'] - batch['pred']))
    print("accuracy_score",accuracy_score(class_true,class_pred))
    print("precision_score",precision_score(class_true,class_pred))
    print("recall_score",recall_score(class_true,class_pred))
    print("f1_score",f1_score(class_true,class_pred))



    fig, axs= plt.subplots( figsize=(18, 12), dpi=100)
    #plt.figure()
    #plt.scatter(pred*100,np.array(y_test)*100)
    plt.plot(np.arange(0, len(pred)*15,15), pred, 'r', label = 'Prediction', linewidth=4)
    plt.plot(np.arange(0, len(y_test)*15,15), y_test, 'b', label = 'Reference', linewidth=4)
    plt.xlim([0,3500])
    plt.ylim([60,105])
    plt.xlabel('Time (s)')
    plt.ylabel(r'SpO$_2$ (%)')
    plt.title("Volunteer " + str(int(vols[i].split('-')[0])%100000))
    plt.tight_layout()
    plt.legend()
    plt.savefig(str(int(vols[i].split('-')[0]))+'.png')


fig_all, axs= plt.subplots( figsize=(24, 24), dpi=100)
for i in range(len(vols)): 
    print("\nVoluntário",vols[i])
    test_vol =  vols[i]
    batch = dnn_results[dnn_results['vol'] == test_vol].copy()
    pred = batch['pred']
    y_test = batch['true']
    #plt.figure()
    #plt.scatter(pred*100,np.array(y_test)*100)
    plt.scatter(pred, y_test, s=200, label = 'vol. '+str(i+1))
    plt.xlim([63,103])
    plt.ylim([63,103])
    plt.xlabel(r'Prediction - SpO$_2$ (%)')
    plt.ylabel(r'Reference - SpO$_2$ (%)')
plt.tight_layout()
plt.legend()
plt.savefig('scatter_all.png')
        





plt.figure(figsize=(18, 12), dpi=100)
plt.scatter(dnn_results['pred'],dnn_results['true'])
plt.xlabel(r'Prediction - SpO$_2$ (%)')
plt.ylabel(r'Reference - SpO$_2$ (%)')
#plt.title("RMSE: {:.2f}".format(rmse)+" || "+"Pearson Coef.: {:.2f}".format(corr))
plt.tight_layout()
#plt.savefig('scatter_all.png')
plt.show()

rmse = np.sqrt(mse(dnn_results['pred'],dnn_results['true']))
corr = np.corrcoef((dnn_results[['pred','true']].values.T))[0,1]
mymae = mae(dnn_results['pred'],dnn_results['true'])

print("\nTotal")
print("RMSE:", rmse)
print("CORRELACAO:", corr)
print("CCC:", ccc(dnn_results['pred'],dnn_results['true']))
print("MAE:", mymae)
print("std", np.std(dnn_results['true'] - dnn_results['pred']))
print("median", np.median(dnn_results['true'] - dnn_results['pred']))

class_true = np.zeros([len(dnn_results),2])
class_true[dnn_results['true']<90,0] = 1
class_true[dnn_results['pred']<90,1] = 1

print(classification_report(y_true = class_true[:,0], y_pred=class_true[:,1]))
print(confusion_matrix(y_true = class_true[:,0], y_pred=class_true[:,1]))


print("accuracy_score",accuracy_score(y_true = class_true[:,0], y_pred=class_true[:,1]))
print("precision_score",precision_score(y_true = class_true[:,0], y_pred=class_true[:,1]))
print("recall_score",recall_score(y_true = class_true[:,0], y_pred=class_true[:,1]))
print("f1_score",f1_score(y_true = class_true[:,0], y_pred=class_true[:,1]))





# RMSE: 4.630561836686834
# CORRELACAO: 0.864793900168815
# CCC: 0.860577987045822
# MAE: 3.833039873316691
# std 4.574069039464401
# median 1.81091337336629

