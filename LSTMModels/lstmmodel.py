import keras
import pickle
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Masking
from keras.layers import Input
from keras.layers import Concatenate

import numpy as np
from matplotlib import pyplot as plt 
from numpy import genfromtxt
from sklearn.utils import shuffle
import csv
from numpy import genfromtxt

import math
import sklearn
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import os 


def create_features(subdir, latency_file, embeddings):
  Features = []
  maxLayer = 0
  maxFlops = 0
  maxDim = 224
  maxChannel = 960
  maxKernel = 7
  maxPadding = 3

  latency = np.genfromtxt(subdir + "/" + latency_file, delimiter=',')
  lat_mean = latency[:,0]
  maxlatency=np.amax(lat_mean)
  lat_mean = lat_mean/maxlatency
  with open(subdir + "/" + embeddings, newline='') as f:
      reader = csv.reader(f)
      data = list(reader)

  for i in range(len(data)):
      temp = [data[i][j * 13:(j + 1) * 13] for j in range((len(data[i]) + 12) // 13 )]
      maxLayer = max(maxLayer, len(temp))
      maxFlops = max(maxFlops, int(temp[:][len(temp)-1][12]))
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


  return maxLayer,lat_mean,numpyFeatures

def learn_lstm_model(hardware, maxLayer, lat_mean, features):
  features, lat_mean = shuffle(features,lat_mean)
  trainf = features[:int(0.85*len(features))]
  trainy = lat_mean[:int(0.85*len(features))]
  testf = features[int(0.85*len(features)):]
  testy = lat_mean[int(0.85*len(features)):]

  #Create an LSTM model
  model=Sequential()
  model.add(Masking(mask_value=-1,input_shape=(maxLayer, 13)))
  model.add(LSTM(20, activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.summary()
  model.fit(trainf, trainy, epochs=800, batch_size=512, verbose=2)

  trainPredict = model.predict(trainf)
  testPredict = model.predict(testf)
  trainScore = math.sqrt(mean_squared_error(trainy, trainPredict[:,0]))
  print('Train Score: %f RMSE' % (trainScore))
  testScore = math.sqrt(mean_squared_error(testy, testPredict[:,0]))
  print('Test Score: %f RMSE' % (testScore))
  plt.xlabel("Actual Latency")
  plt.ylabel("Predicted Latency")
  plt.scatter(testy, testPredict[:,0])
  plt.savefig(hardware+'.png')
  #plt.show()
  print("The R^2 Value for %s:"%(hardware), sklearn.metrics.r2_score(testy, testPredict[:,0]))




def main():
  list_val_dict = {}
  execTime = []
  embeddings = []
  val = False
  for subdir, dirs, files in os.walk(os.getcwd()):
    for file in files:
      if file == "execTime.csv":
        execTime = file
      elif file == "Embeddings.csv":
        embeddings = file
        val = True

    if val==True:
      print(execTime, embeddings)
      print(subdir)
      list_val_dict[os.path.basename(subdir)] = create_features(subdir,execTime,embeddings)
      val = False
      #print(os.path.basename(subdir), file)

  for key in list_val_dict:
    learn_lstm_model(key, list_val_dict[key][0], list_val_dict[key][1], list_val_dict[key][2])

if __name__ == '__main__':
  main()
	


