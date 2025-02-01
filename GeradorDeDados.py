# -*- coding: utf-8 -*-
"""
Data generator referrence:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""

import numpy as np
from tensorflow import keras
import pandas as pd

class DataGenerator(keras.utils.Sequence):

  def __init__(self, data_list, length, shuffle=False, batch_size=10, n_batches=1024):
    """
    Initializes a data generator object
      :param csv_file: file in which image names and numeric labels are stored
      :param base_dir: the directory in which all images are stored
      :param output_size: image output size after preprocessing
      :param shuffle: shuffle the data after each epoch
      :param batch_size: The size of each batch returned by __getitem__
    """
    self.data_list = data_list
    self.length = length
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.on_epoch_end()
    #
    self.n_batches = n_batches

  def on_epoch_end(self):
      pass
    # self.indices = np.arange(len(self.df))
    # if self.shuffle:
    #   np.random.shuffle(self.indices)

  def __len__(self):
    return self.n_batches
    #return int(len(self.df) / self.batch_size)

  def __getitem__(self, idx):
    ## Initializing Batch
    #  that one in the shape is just for a one channel images
    # if you want to use colored images you might want to set that to 3
    X = np.empty((self.batch_size, self.length, 2))
    # (x, y, h, w)
    y = np.empty((self.batch_size, 1))

    # get the indices of the requested batch
    #indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

    for i in range(self.batch_size):
      #selecionar um voluntario aleateriamente
      rand_vol = np.random.randint(low=0, high=len(self.data_list))
      sel_data = self.data_list[rand_vol].copy()
      
      # pegar um indice aleatorio
      idx = np.random.randint(low = 0, high = len(sel_data)-self.length)
      
      X[i,] = sel_data[['R','G']].iloc[idx:idx+self.length].values
      y[i] = sel_data['S'].iloc[idx:idx+self.length].mean()

    return X, y

# test
def main():
    import matplotlib.pyplot as plt
    import os
    
    data_path = "dados"
    
    data_batch = []
    
    files = os.listdir(data_path)
    
    for fl in files:
        data_batch.append(pd.read_csv(os.path.join(data_path,fl), index_col=0))    
    
    dg = DataGenerator(data_list = data_batch, length=256, 
                                    shuffle=False, batch_size=1024, n_batches=10)
    
    # teste do gerador
    X, y = dg.__getitem__(0)
    print('X shape:',X.shape)
    print('y shape:',y.shape)
    
    print(dg.__len__())
    
if __name__ == '__main__':
    main()