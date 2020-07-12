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
import multiprocessing as mp
import matplotlib.cm
from lstmmodel import *
numLatency = 1000
def main():
    list_val_dict = {}
    execTime = []
    embeddings = "Embeddings.csv"
    val = False
    for subdir, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file == "execTime.csv":
                execTime = file
                val = True
            #elif file == "Embeddings.csv":
            #    embeddings = file
            #    val = True
        if val==True:
            print(execTime, embeddings)
            print(subdir)
            tmp_list = []
            maxLayer, lat_mean, numFeatures = parse_features(subdir, execTime, embeddings)
            tmp_list.append(maxLayer)
            tmp_list.append(lat_mean)
            tmp_list.append(numFeatures)
            print(numFeatures.shape, tmp_list[2].shape)
            list_val_dict[os.path.basename(subdir)] = tmp_list
            val = False

    numHardware = len(list_val_dict)
    hardwareList = list(list_val_dict.keys())
    transferR2Score = np.zeros((numHardware, numHardware))
    transferSpearman = np.zeros((numHardware, numHardware))
    for i in range(numHardware):
        key = hardwareList[i]
        model = learn_lstm_model(key, list_val_dict[key][0], list_val_dict[key][1], list_val_dict[key][2], 13)
        for j in range(numHardware):
            if j == i:
                continue
            transferKey = hardwareList[j]
            features, lat = shuffle(list_val_dict[transferKey][2], list_val_dict[transferKey][1])
            trainf = features[:int(0.05*len(features))]
            trainy = lat[:int(0.05*len(lat))]
            testf = features[int(0.05*len(features)):]
            testy = lat[int(0.05*len(features)):]


            trainPredict = model.predict(trainf)
            testPredict = model.predict(testf)
            trainScore = math.sqrt(mean_squared_error(trainy, trainPredict))
            print('Train Score: %f RMSE' % (trainScore))
            testScore = math.sqrt(mean_squared_error(testy, testPredict))
            print('Test Score: %f RMSE' % (testScore))
            r2_score = sklearn.metrics.r2_score(testy, testPredict)
            s_coefficient, pvalue = spearmanr(testy, testPredict)
            print("The transferred R^2 Value for %s:"%(transferKey), r2_score)
            print("The transferred Spearnman Coefficient and p-value for %s: %f and %f"%(transferKey, s_coefficient, pvalue))

            transferR2Score[i][j] = r2_score
            transferSpearman[i][j] = s_coefficient

    np.savetxt("transferR2score.csv", transferR2Score, delimiter=",")
    np.savetxt("transferSpearMan.csv", transferSpearman, delimiter=",")
    print(hardwareList)
if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    main()
