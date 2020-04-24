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
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import os
import multiprocessing as mp
import matplotlib.cm
from lstmmodel import *

def plotLatnecyRandomSamples(list_val_dict):
    ## ------------------Random Sampling------------------
    maxSamples = 30
    final_indices = random_indices(maxSamples)
    latency = np.zeros((len(list_val_dict), maxSamples))
    i = 0
    for key in list_val_dict:
        for j in range(maxSamples):
            latency[i][j] = list_val_dict[key][1][final_indices[j]]
        i+=1
    
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, latency.shape[0]))
    labels = list(list_val_dict.keys())

    fig, ax = plt.subplots()
    ax.set_title("RandomSampling")
    ax.set_xlabel("networks")
    ax.set_ylabel("Latency")
    for i in range(latency.shape[0]):
        ax.scatter(np.arange(latency.shape[1]), latency[i][:], color=colors[i], label=labels[i])
    # Shrink current axis by 20%
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.legend()
    fig.savefig("RandomSampling.png")

def plotTSNERandomSamples(list_val_dict):
    ## ------------------Random Sampling------------------
    maxSamples = 30
    final_indices = random_indices(maxSamples)
    latency = np.zeros((len(list_val_dict), maxSamples))
    i = 0
    for key in list_val_dict:
        for j in range(maxSamples):
            latency[i][j] = list_val_dict[key][1][final_indices[j]]
        i+=1
    
    for i in range(latency.shape[0]):
        temp = np.reshape(latency[i][:], (-1,1))
        transformedOut = TSNE(n_components=2).fit_transform(temp)
        print(transformedOut.shape)
        plt.scatter(transformedOut[:][0], transformedOut[:][1])
        plt.show()

def plotLatnecyStatSamples(list_val_dict):
    #----------Statistical Sampling With HoldOut Hardwares-------------------
    for key in list_val_dict:
        list_val_dict_local = copy.deepcopy(list_val_dict)
        hold_out_val = list_val_dict_local[key]
        hold_out_key = key
        list_val_dict_local.pop(key)
        
        final_indices, hw_features_cncat = sample_hwrepresentation(list_val_dict_local, 30)
        maxSamples = len(final_indices)
        latency = np.zeros((len(list_val_dict_local), maxSamples))
        i = 0
        for key in list_val_dict_local:
            for j in range(maxSamples):
                latency[i][j] = list_val_dict[key][1][final_indices[j]]
            i+=1
        
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, latency.shape[0]))
        labels = list(list_val_dict_local.keys())

        fig, ax = plt.subplots()
        ax.set_title("Statistical_Without_"+str(hold_out_key))
        ax.set_xlabel("Networks")
        ax.set_ylabel("Latency")
        for i in range(latency.shape[0]):
            ax.scatter(np.arange(latency.shape[1]), latency[i][:], color=colors[i], label=labels[i])
        ax.legend()
        fig.savefig("Statistical_Without_"+str(hold_out_key)+".png")

def plotLatnecyMISamples(list_val_dict):
    ##---------Mutual Information 1--------------------
    for key in list_val_dict:
        list_val_dict_local = copy.deepcopy(list_val_dict)
        hold_out_val = list_val_dict_local[key]
        hold_out_key = key
        list_val_dict_local.pop(key)
        
        final_indices, hw_features_cncat = mutual_information_v2(list_val_dict_local, 30)
        maxSamples = len(final_indices)
        latency = np.zeros((len(list_val_dict_local), maxSamples))
        i = 0
        for key in list_val_dict_local:
            for j in range(maxSamples):
                latency[i][j] = list_val_dict[key][1][final_indices[j]]
            i+=1
        
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, latency.shape[0]))
        labels = list(list_val_dict_local.keys())

        fig, ax = plt.subplots()
        ax.set_title("MutualInfo_Without_"+str(hold_out_key))
        ax.set_xlabel("Networks")
        ax.set_ylabel("Latency")
        for i in range(latency.shape[0]):
            ax.scatter(np.arange(latency.shape[1]), latency[i][:], color=colors[i], label=labels[i])
        ax.legend()
        fig.savefig("MutualInfo_Without_"+str(hold_out_key)+".png")

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
            tmp_list = []
            maxLayer, lat_mean, numFeatures = parse_features(subdir, execTime, embeddings)
            tmp_list.append(maxLayer)
            tmp_list.append(lat_mean)
            tmp_list.append(numFeatures)
            print(numFeatures.shape, tmp_list[2].shape)
            list_val_dict[os.path.basename(subdir)] = tmp_list
            val = False

    # plotLatnecyRandomSamples(list_val_dict)
    # plotLatnecyStatSamples(list_val_dict)
    # plotLatnecyMISamples(list_val_dict)
    plotTSNERandomSamples(list_val_dict)
if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    main()

