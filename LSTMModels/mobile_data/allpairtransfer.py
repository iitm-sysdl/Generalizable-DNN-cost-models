import keras
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Masking
from keras.layers import Input
from keras.layers import Concatenate
from keras import optimizers
from scipy.stats import spearmanr
import copy
import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
from sklearn.utils import shuffle
import csv
import random
import math
import sklearn
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import os
import glob
import random
def parse_latency(file):
    data = np.genfromtxt(file, delimiter=',')
    latency = np.mean(data, axis=1)
    latency = latency[:100]
    latency = latency/np.amax(latency)
    return latency

def parse_features():
  Features = []
  maxLayer = 0
  maxFlops = 0
  maxChannel = 0
  maxDim = 224
  maxKernel = 7
  maxPadding = 3

  with open('Embeddings.csv', newline='') as f:
      reader = csv.reader(f)
      data = list(reader)

  for i in range(len(data)):
      temp = [data[i][j * 13:(j + 1) * 13] for j in range((len(data[i]) + 12) // 13 )]
      maxLayer = max(maxLayer, len(temp))
      for j in range(len(temp)):
            maxFlops=max(maxFlops, float(temp[j][12]))
            maxChannel = max(maxChannel, int(temp[j][7]))
            maxChannel = max(maxChannel, int(temp[j][8]))
      Features.append(temp)

  numpyFeatures = np.ones((len(Features), maxLayer, 13))
  numpyFeatures = numpyFeatures*-1

  for i in range(len(Features)):
    temp = Features[i]
    for j in range(len(temp)):
      for k in range(len(temp[j])):
        numpyFeatures[i][j][k] = temp[j][k]
        if k == 5 or k == 6:
          numpyFeatures[i][j][k] = numpyFeatures[i][j][k]/maxDim
        elif k == 7 or k == 8:
          numpyFeatures[i][j][k] = numpyFeatures[i][j][k]/maxChannel
        elif k == 9:
          numpyFeatures[i][j][k] = numpyFeatures[i][j][k]/maxKernel
        elif k == 12:
          numpyFeatures[i][j][k] = numpyFeatures[i][j][k]/maxFlops

  return numpyFeatures, maxLayer

def learn_lstm_model(hardware, maxLayer, lat_mean, features, featuresShape):
  numSample = len(lat_mean)
  features = features[:numSample]
  features, lat_mean = shuffle(features,lat_mean)
  trainf = features[:int(0.9*len(features))]
  trainy = lat_mean[:int(0.9*len(features))]
  testf = features[int(0.9*len(features)):]
  testy = lat_mean[int(0.9*len(features)):]
  print(trainf.shape, trainy.shape, testf.shape, testy.shape)

  #Create an LSTM model
  model=Sequential()
  model.add(Masking(mask_value=-1,input_shape=(maxLayer, featuresShape)))
  model.add(LSTM(20, activation='relu'))
  model.add(Dense(1))
  '''Adam intialized with Default Values. Tune only intial Learning rate.
  opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)'''
  opt = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
  model.compile(loss='mean_squared_error', optimizer=opt)
  model.summary()
  model.fit(trainf, trainy, epochs=400, batch_size=32, verbose=1)

  trainPredict = model.predict(trainf)
  testPredict = model.predict(testf)
  trainScore = math.sqrt(mean_squared_error(trainy, trainPredict))
  print('Train Score: %f RMSE' % (trainScore))
  testScore = math.sqrt(mean_squared_error(testy, testPredict))
  r2_score = sklearn.metrics.r2_score(testy, testPredict)
  s_coefficient, pvalue = spearmanr(testy, testPredict)
  print('Test Score: %f RMSE' % (testScore))
  print("The R^2 Value for %s:"%(hardware), r2_score)
  print("The Spearnman Coefficient and p-value for %s: %f and %f"%(hardware, s_coefficient, pvalue))

  plt.figure()
  plt.xlabel("Actual Latency")
  plt.ylabel("Predicted Latency")
  plt.scatter(testy, testPredict[:,0])
  plt.title(hardware+' PearR2: '+str(r2_score)+' SpearR2: '+str(s_coefficient))
  plt.savefig(hardware+'.png')
  #plt.show()
  return model


def main():
    features, maxLayers = parse_features()

    files = glob.glob('*.txt')    
    holdoutHW = random.choice(files)
    latency = parse_latency(holdoutHW)

    model = learn_lstm_model(holdoutHW, maxLayers, latency, features, 13)
    #exit()   

    for file in files:
        if file == holdoutHW:
            continue
        transferKey = file
        lat = parse_latency(file)
        features, lat = shuffle(features, lat)

        predict = model.predict(features)
        transferScore = math.sqrt(mean_squared_error(predict, lat))
        print('Transfer Score on' + file + ' : %f RMSE' % (transferScore))
        r2_score = sklearn.metrics.r2_score(lat, predict)
        s_coefficient, pvalue = spearmanr(lat, predict)
        print("The transferred R^2 Value for %s:"%(transferKey), r2_score)
        print("The transferred Spearnman Coefficient^2 and p-value for %s: %f and %f"%(transferKey, s_coefficient**2, pvalue))
        plt.scatter(lat, predict)
        plt.title(transferKey+' PearR2: '+str(r2_score)+' SpearR2: '+str(s_coefficient**2))
        plt.savefig(transferKey+'.png')

if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    main()
