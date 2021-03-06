'''
Authors: Vinod Ganesan, Surya Selvam, Pratyush Kumar
Email ID: vinodg@cse.iitm.ac.in, selvams@purdue.edu, pratyush@cse.iitm.ac.in
'''

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
#from tensorflow.keras import layers
#from tensorflow.keras import optimizers
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from sklearn.neighbors import KernelDensity
from keras.layers import Masking
from keras.layers import Input
from keras.layers import Concatenate
from keras import optimizers
from scipy.stats import spearmanr
from scipy import stats
from statistics import mean
import copy
import mlflow
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt
from sklearn.utils import shuffle
import csv
import random
import math
import sklearn
import mlflow
import mlflow.keras
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import os
import glob
import multiprocessing as mp
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.cm
import matplotlib
import argparse
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from xgboost import XGBRFRegressor
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
numLatency = 118
embeddingsFile = "onnxEmbeddings.csv"
lat = []
maxVal = 0
matplotlib.use('Agg')
def parse_latency(file):
    global lat
    data = np.genfromtxt(file, delimiter=',')
    latency = np.mean(data, axis=1)
    latency = latency[:numLatency]
    lat.append(latency)
    #latency = latency/np.amax(latency)
    return latency

def parse_features():
  Features = []
  maxLayer = 0
  maxFlops = 0
  maxChannel = 0
  maxDim = 224
  maxKernel = 7
  maxPadding = 3

  with open(embeddingsFile, newline='') as f:
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

def learn_xgb_model(hardware, maxLayer, lat_mean, features, featuresShape, splitPercentage=0.99, shuffleFeatures=True):
  numSample = len(lat_mean)
  features = features[:numSample]
  if shuffleFeatures == True:
    features, lat_mean = shuffle(features,lat_mean)
  trainf = features[:int(splitPercentage*len(features))]
  trainy = lat_mean[:int(splitPercentage*len(features))]
  testf = features[int(splitPercentage*len(features)):]
  testy = lat_mean[int(splitPercentage*len(features)):]
  print("================= Dataset Stage ==============")
  print(trainf.shape, trainy.shape, testf.shape, testy.shape)
  trainf = np.reshape(trainf, (trainf.shape[0], trainf.shape[1]*trainf.shape[2]))
  testf  = np.reshape(testf, (testf.shape[0], testf.shape[1]*testf.shape[2]))

  model = XGBRegressor()
  model.fit(trainf, trainy)

  trainPredict = model.predict(trainf)
  testPredict = model.predict(testf)
  trainScore = math.sqrt(mean_squared_error(trainy, trainPredict))
  testScore = math.sqrt(mean_squared_error(testy, testPredict))

  ### Train Model characteristics
  r2_score = sklearn.metrics.r2_score(trainy, trainPredict)
  s_coefficient, pvalue = spearmanr(trainy, trainPredict)
  writeToFile('Train Score: %f RMSE' % (trainScore))
  writeToFile("The R^2 Value for %s: %f"%(hardware, r2_score))
  writeToFile("The Spearnman Coefficient and p-value for %s: %f and %f"%(hardware, s_coefficient, pvalue))

  plt.figure()
  plt.xlabel("Actual Latency (in ms)")
  plt.ylabel("Predicted Latency (in ms)")
  sns.scatterplot(trainy, trainPredict)
  plt.savefig(args.name+'/plots/'+hardware+'_'+args.learning_type+'_'+str(splitPercentage)+'_train.png')

  r2_score = sklearn.metrics.r2_score(testy, testPredict)
  s_coefficient, pvalue = spearmanr(testy, testPredict)
  writeToFile('Test Score: %f RMSE' % (testScore))
  writeToFile("The R^2 Value for %s: %f"%(hardware, r2_score))
  writeToFile("The Spearnman Coefficient and p-value for %s: %f and %f"%(hardware, s_coefficient, pvalue))

  plt.figure()
  plt.xlabel("Actual Latency (in ms)")
  plt.ylabel("Predicted Latency (in ms)")
  sns.scatterplot(testy, testPredict)
  plt.savefig(args.name+'/plots/'+hardware+"_"+args.learning_type+'_'+str(1-splitPercentage)+'_test.png')
  return model

def learn_xgb_model_collab(hardware, maxLayer, lat_mean, features, featuresShape, splitPercentage=0.99, shuffleFeatures=True):
  print('Learning' + hardware)
  numSample = len(lat_mean)
  features = features[:numSample]
  if shuffleFeatures == True:
    features, lat_mean = shuffle(features,lat_mean)
  testf = features
  testy = lat_mean
  testf  = np.reshape(testf, (testf.shape[0], testf.shape[1]*testf.shape[2]))
  results = []
  index = []
  for i in range(10, numSample):
    trainf = features[:i]
    trainy = lat_mean[:i]
#   print("================= Dataset Stage ==============")
#   print(trainf.shape, trainy.shape, testf.shape, testy.shape)
    trainf = np.reshape(trainf, (trainf.shape[0], trainf.shape[1]*trainf.shape[2]))
    model = XGBRegressor()
    model.fit(trainf, trainy)
    testPredict = model.predict(testf)
    testScore = math.sqrt(mean_squared_error(testy, testPredict))

    r2_score = sklearn.metrics.r2_score(testy, testPredict)
    s_coefficient, pvalue = spearmanr(testy, testPredict)
    results.append(r2_score)
    index.append(i)

  matplotlib.rcParams['figure.dpi'] = 500
  plt.figure()
  plt.xlabel("Number of Datapoints")
  plt.ylabel("Average R^2")
  sns.lineplot(index, results)
  plt.savefig(args.name+'/plots/'+hardware+'_indiLearn.png')
  f = open(args.name+'/meta/plotdata.txt', a)
  s1 = ','.join(map(str, index))
  s2 = ','.join(map(str, results))
  f.write(hardware+'\n'+s1+'\n'+s2+'\n')
  f.close()

def learn_lstm_model(hardware, maxLayer, lat_mean, features, featuresShape):
  numSample = len(lat_mean)
  features = features[:numSample]
  features, lat_mean = shuffle(features,lat_mean)
  trainf = features[:int(0.99*len(features))]
  trainy = lat_mean[:int(0.99*len(features))]
  #testf = features[:int(1.0*len(features))]
  #testy = lat_mean[:int(1.0*len(features))]
  testf = features[int(0.99*len(features)):]
  testy = lat_mean[int(0.99*len(features)):]
  print("================= Dataset Stage ==============")
  print(trainf.shape, trainy.shape, testf.shape, testy.shape)

  #mlflow.keras.autolog()

  #Create an LSTM model
  model=Sequential()
  model.add(Masking(mask_value=-1,input_shape=(maxLayer, featuresShape)))
  model.add(LSTM(20, activation='relu'))
  model.add(Dense(1, name = 'fc'))
  opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

  #initial_learning_rate = 0.01

 # lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate,

  #opt = optimizers.SGD(learning_rate = initial_learning_rate)
  model.compile(loss='mean_squared_error', optimizer=opt, metrics=[keras.metrics.MeanAbsolutePercentageError()])
  model.summary()
  #filepath="checkpoint-{loss:.5f}-{val_loss:.5f}-{val_mean_absolute_percentage_error}.hdf5"
  filepath=args.name+'/models/model.hdf5'
  #checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')#montor can be val_loss or loss
  checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')#montor can be val_loss or loss
  es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=50)
  val = model.fit(trainf, trainy, epochs=250, batch_size=512, verbose=1, callbacks=[es, checkpoint])
  #val = model.fit(trainf, trainy, epochs=250, batch_size=512, verbose=1, callbacks=[es, checkpoint], validation_data=(testf, testy))
  model.load_weights(filepath)

  trainPredict = model.predict(trainf)
  testPredict = model.predict(testf)
  trainScore = math.sqrt(mean_squared_error(trainy, trainPredict))
  writeToFile('Train Score: %f RMSE' % (trainScore))
  testScore = math.sqrt(mean_squared_error(testy, testPredict))

  ### Train Model characteristics
  r2_score = sklearn.metrics.r2_score(trainy, trainPredict)
  s_coefficient, pvalue = spearmanr(trainy, trainPredict)
  writeToFile('Train Score: %f RMSE' % (trainScore))
  writeToFile("The R^2 Value for %s: %f"%(hardware, r2_score))
  writeToFile("The Spearnman Coefficient and p-value for %s: %f and %f"%(hardware, s_coefficient, pvalue))

  plt.figure()
  plt.xlabel("Actual Latency (in ms)")
  plt.ylabel("Predicted Latency (in ms)")
  sns.scatterplot(trainy, trainPredict[:,0])
  #plt.title(hardware+' R2: '+str(r2_score)+' SpearVal: '+str(s_coefficient))
  plt.savefig(args.name+'/plots/'+hardware+"_"+args.learning_type+'_train.png')


  r2_score = sklearn.metrics.r2_score(testy, testPredict)
  s_coefficient, pvalue = spearmanr(testy, testPredict)
  writeToFile('Test Score: %f RMSE' % (testScore))
  writeToFile("The R^2 Value for %s: %f"%(hardware, r2_score))
  writeToFile("The Spearnman Coefficient and p-value for %s: %f and %f"%(hardware, s_coefficient, pvalue))

  plt.figure()
  plt.xlabel("Actual Latency (in ms)")
  plt.ylabel("Predicted Latency (in ms)")
  sns.scatterplot(testy, testPredict[:,0])
  #plt.title(hardware+' R2: '+str(r2_score)+' SpearVal: '+str(s_coefficient))
  plt.savefig(args.name+'/plots/'+hardware+"_"+args.learning_type+'_test.png')

### Adding Other Regressors
  extractor = Model(outputs=model.get_layer('fc').input, inputs=model.input)
  extractor.summary()
  knn = KNeighborsRegressor()
  trainPredict = extractor.predict(trainf)
  testPredict = extractor.predict(testf)
  randForest = RandomForestRegressor()
  decisionTree = DecisionTreeRegressor()
  svr = SVR()
  kernelrdidge = KernelRidge()
  xgb = XGBRegressor()
  xgbrf = XGBRFRegressor()
  modellist = [ ('knn', knn), ('randomForest', randForest), ('dTree', decisionTree), ('svr', svr), ('kerenlrdige', kernelrdidge), ('xgb', xgb), ('xgbrf', xgbrf) ]
  for name, model_lowB in modellist:
    model_lowB.fit(trainPredict, trainy)
    modeltestPred = model_lowB.predict(testPredict)
    testScore = math.sqrt(mean_squared_error(testy, modeltestPred))
    r2_score = sklearn.metrics.r2_score(testy, modeltestPred)
    s_coefficient, pvalue = spearmanr(testy, modeltestPred)
    writeToFile('Test Score with %s : %f RMSE' % (name, testScore))
    writeToFile("The R^2 Value with %s for %s: %f"%(hardware, name, r2_score))
    writeToFile("The Spearnman Coefficient and p-value for %s with %s : %f and %f"%(hardware, name, s_coefficient, pvalue))
    plt.figure()
    plt.xlabel("Actual Latency (in ms)")
    plt.ylabel("Predicted Latency (in ms)")
    sns.scatterplot(testy, modeltestPred)
    #plt.title(name + hardware+' R2: '+str(r2_score)+' SpearVal: '+str(s_coefficient))
    plt.savefig(args.name+'/plots/'+hardware+args.learning_type+'_'+name+'.png')
  return (model, modellist, extractor)


'''
This function takes in the dictionary of hardware_names to its maxLayer, latency and features map
net_dict[key][2] - refers to the network features for a hardware and
net_dict[key][1] - refers to the latency for that hardware

1. First determine the mean and std of the latencies for each hardware in the dictionary

2. Sample from the distribution - i.e. from Mu-8*sigma to Mu+2*sigma, at each parts of the distribution, find all indices that intersect in all the hardwares considered here. For ex., if network no. 2374 falls between mu-1*sigma and mu for all the hardware devices in the dictionary, then add 2374 to the representation set for all the hardware

3. Find maxSamples such networks that become the golden representation of the hardware

4. Return the list of lists of maxSamples network representation for all hardwares and also the indices of the representation networks

5. The indices will be used by any hardware not on the list to make and append it's representation
TODO: Not using max samples for now - change
'''

def sample_hwrepresentation(net_dict, maxSamples):
    mean_lat = []
    sd_lat = []
    final_indices = []
    #Determining the Mean and Standard Deviation of Latencies
    for key in net_dict:
        net_dict[key][2] = net_dict[key][2][:numLatency,:,:] #Not required actually.. Simply doing
        net_dict[key][1] = net_dict[key][1][:numLatency]
        print(np.mean(net_dict[key][1]), np.std(net_dict[key][1]))
        mean_lat.append(np.mean(net_dict[key][1]))
        sd_lat.append(np.std(net_dict[key][1]))
    for i in range(-2,8): #This range might not be enough -- the range should be more generic when hardware increases
        index_where = []
        index = 0

        for key in net_dict:
            index_where.append(np.where(np.logical_and(net_dict[key][1] > mean_lat[index]+i*sd_lat[index], net_dict[key][1] <= mean_lat[index]+(i+1)*sd_lat[index])))
            index += 1
        for j in range(len(index_where)):
            index_where[0] = np.intersect1d(index_where[0], index_where[j])

        final_intersection = index_where[0]
        if len(final_intersection) >= 4:
            loop_index = 4
        else:
            loop_index = len(final_intersection)

        hw_features_cncat = []

        for j in range(loop_index):
            final_indices.append(final_intersection[j])

    print("The final indices size is %f"%(len(final_indices)))

    for key in net_dict:
        hw_features_per_device = []
        for j in range(len(final_indices)):
            hw_features_per_device.append(net_dict[key][1][final_indices[j]])
        net_dict[key][1] = np.delete(net_dict[key][1], final_indices, axis=0)
        net_dict[key][2] = np.delete(net_dict[key][2], final_indices, axis=0)
        hw_features_cncat.append(hw_features_per_device)
    print(len(final_indices), net_dict[key][2].shape)
    return final_indices, hw_features_cncat


def random_indices(maxSamples):
    rand_indices = []
    for i in range(maxSamples):
        rand_indices.append(random.randint(0,numLatency-1))
    return rand_indices
'''
Function which computes total MACs of each network and samples maxSamples indices from it based on FLOPS.
'''
def flopsBasedIndices(maxSamples):
    with open('../DiverseRandNetworkGenerator/Embeddings.csv') as f:
        reader = csv.reader(f)
        data = list(reader)

    totalFLOPSList = np.zeros(len(data))
    for i in range(len(data)):
        temp = [data[i][j * 13:(j + 1) * 13] for j in range((len(data[i]) + 12) // 13 )]
        for j in range(len(temp)):
            totalFLOPSList[i]+=int(temp[j][12])

    mean = np.mean(totalFLOPSList)
    sd = np.std(totalFLOPSList)

def random_sampling(net_dict, rand_indices, maxSamples):
    for key in net_dict:
        net_dict[key][2] = net_dict[key][2][:numLatency,:,:]
        net_dict[key][1] = net_dict[key][1][:numLatency]

    hw_features_cncat = []
    #rand_indices = []
    #final_indices = []

    #for i in range(maxSamples):
    #    rand_indices.append(random.randint(0,5000))

    for key in net_dict:
        hw_features_per_device = []
        for j in range(maxSamples):
            hw_features_per_device.append(net_dict[key][1][rand_indices[j]])
        hw_features_cncat.append(hw_features_per_device)

    #If this is not done separately, the code will break
    for key in net_dict:
        net_dict[key][1] = np.delete(net_dict[key][1], rand_indices, axis=0)
        net_dict[key][2] = np.delete(net_dict[key][2], rand_indices, axis=0)



    return hw_features_cncat



'''
Append the hardware representation with the available network representation in axis = 2 (3rd dimension)
and also append all the hardwares together along axis = 0 (row dimension) to form a huge training set of multiple
hardware devices
'''

def append_with_net_features(net_dict, hw_features_cncat):
    new_lat_ft = []
    appended_features = []
    appended_latencies = []
    index = 0
    for key in net_dict:
        print("======================================================")
        print(len(hw_features_cncat[index]))
        new_lat_ft = np.tile(hw_features_cncat[index], (net_dict[key][2].shape[0], net_dict[key][2].shape[1], 1))
        temp = np.concatenate((net_dict[key][2], new_lat_ft), axis=2)
        print(new_lat_ft.shape, net_dict[key][2].shape, temp.shape)
        if index == 0:
            appended_features = temp
            appended_latencies = net_dict[key][1]
        else:
            appended_features = np.concatenate((appended_features, temp), axis=0)
            appended_latencies = np.concatenate((appended_latencies, net_dict[key][1]), axis=0)
        index += 1
        print(appended_features.shape, appended_latencies.shape)
        #print(appended_features, appended_latencies)
    return appended_latencies, appended_features


def corr_choose(rho, maxSamples, threshold = 0.97, stop_condition = 5, debug=True):
    elements = [numLatency]
    subset = []
    indices = range(rho.shape[0])
    if debug:
        print("Before start : Number of remaining vectors", rho.shape[0])
    for i in range(maxSamples):
        # add_ = np.argmax(np.sum(rho, axis=1))
        add_ = np.argmax(np.sum(rho > threshold, axis=1))
        subset += [indices[add_]]
        remove_set = []
        for j in range(rho.shape[0]):
            if rho[j, add_] > threshold:
                remove_set += [j]
        rho = np.delete(rho, remove_set, axis=0)
        rho = np.delete(rho, remove_set, axis=1)
        indices = np.delete(indices, remove_set)
        elements.append(rho.shape[0])
        if debug:
            print('Iteration', i, ": Number of remaining vectors", rho.shape[0])
        if len(indices) <= stop_condition:
            break
    if debug:
        print('Chosen networks are ', subset)

    matplotlib.rcParams['figure.dpi'] = 500
    plt.figure()
    plt.xlabel('Iterations')
    plt.ylabel('Number of Networks remaining')
    # plt.title('Mutual Information Score over iterations')
    plt.plot(np.arange(len(elements)), elements,'-o')
    plt.savefig(args.name+'/plots/spearman.png')
    return subset

def corr_eval(rho, subset, threshold = 0.97):
    count_close = 0
    for i in range(rho.shape[0]):
        if i in subset:
            count_close += 1
            continue
        max_ = 0
        for j in subset:
            max_ = max(rho[i, j], max_)
        if max_ > threshold:
            count_close += 1
    return count_close/rho.shape[0]


def spearmanCorr(net_dict, numSamples):
    index = 0
    global lat
    ll = np.array(lat)
    for key in net_dict:
        net_dict[key][2] = net_dict[key][2][:numLatency, :, :]
        net_dict[key][1] = net_dict[key][1][:numLatency]

    for key in net_dict:
        if index == 0:
            stacked_arr = net_dict[key][1]
        else:
            stacked_arr = np.column_stack((stacked_arr, net_dict[key][1]))
        index+=1

    rho, p = spearmanr(ll)

    print(rho)

    print(rho.shape)

    sel_list = corr_choose(rho, numSamples, 0.98)
    print('Evaluation scores is', corr_eval(rho, sel_list, 0.98))
    #exit(0)

    hw_features_cncat = []

    for key in net_dict:
        hw_features_per_device = []
        for j in range(len(sel_list)):
            hw_features_per_device.append(net_dict[key][1][sel_list[j]])
        hw_features_cncat.append(hw_features_per_device)

    #If this is not done separately, the code will break
    for key in net_dict:
        net_dict[key][1] = np.delete(net_dict[key][1], sel_list, axis=0)
        net_dict[key][2] = np.delete(net_dict[key][2], sel_list, axis=0)

    return sel_list, hw_features_cncat

def pearsonCorr(net_dict, numSamples):
    index = 0
    global lat
    ll = np.array(lat)
    for key in net_dict:
        net_dict[key][2] = net_dict[key][2][:numLatency, :, :]
        net_dict[key][1] = net_dict[key][1][:numLatency]

    for key in net_dict:
        if index == 0:
            stacked_arr = net_dict[key][1]
        else:
            stacked_arr = np.column_stack((stacked_arr, net_dict[key][1]))
        index+=1

    rho = np.corrcoef(ll)

    print(rho)

    print(rho.shape)

    sel_list = corr_choose(rho, numSamples, 0.98)
    print('Evaluation scores is', corr_eval(rho, sel_list, 0.98))

    #exit(0)

    hw_features_cncat = []

    for key in net_dict:
        hw_features_per_device = []
        for j in range(len(sel_list)):
            hw_features_per_device.append(net_dict[key][1][sel_list[j]])
        hw_features_cncat.append(hw_features_per_device)

    #If this is not done separately, the code will break
    for key in net_dict:
        net_dict[key][1] = np.delete(net_dict[key][1], sel_list, axis=0)
        net_dict[key][2] = np.delete(net_dict[key][2], sel_list, axis=0)

    return sel_list, hw_features_cncat


def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def chooseFirstNetMI(data):
    kde = np.ones_like(data)
    print(data.shape)
    for i in range(data.shape[0]):
        a = data[i].reshape(-1,1)
        # print(a.shape)
        k = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(a)
        kde[i] = k.score_samples(a) #sample(a.shape[0])
        kde[i] = np.exp(kde[i])
    print(kde.shape)
    meanval = np.mean(kde, axis=0)
    print(meanval.shape)
    print(meanval)
    maxval = -10000000
    maxindex = 0
    for i in range(kde.shape[0]):
        val = KL(meanval, kde[i])
        print(val)
        if val >= maxval:
            maxval = val
            maxindex = i
    return maxindex

## Mutual Information Implementation

def mutual_information_v2(net_dict, numSamples, choose_minimal=True):
    index = 0
    ## Rows - Networks, Columns - Hardware

    for key in net_dict:
        net_dict[key][2] = net_dict[key][2][:numLatency,:,:]
        net_dict[key][1] = net_dict[key][1][:numLatency]

    for key in net_dict:
        if index == 0:
            stacked_arr = net_dict[key][1]
        else:
            stacked_arr = np.column_stack((stacked_arr, net_dict[key][1]))
        index+=1

    quantize = np.arange(0, 101, 33)
    nlevels = len(quantize)
    print(stacked_arr.shape)
    nrows = stacked_arr.shape[0]
    ncols = stacked_arr.shape[1]

    for i in range(nrows):
        a_ = stacked_arr[i, :]
        p = np.percentile(a_, quantize)
        bins = np.digitize(a_, p)
        stacked_arr[i, :] = bins - 1
    # print(stacked_arr[0:5,:])
    # exit()
    #val = np.random.randint(0, nrows)
    #val = select_network()

    val = chooseFirstNetMI(stacked_arr)

    sel_list = [val]
    hw_features_cncat = []
    max_info_lst = []
    print( " ------------------------------------- Beginning Sampling -------------------")
    for k in range(numSamples-1):
        max_info = 0
        for i in range(nrows):
            if i in sel_list:
                continue
            m = -1*mutual_info(stacked_arr, sel_list + [i], nrows, ncols)

            if m >= max_info:
                max_index = i
                max_info = m
        max_info_lst.append(max_info)
        sel_list = sel_list + [max_index]


    print(" ------------------------------- Done Sampling -----------------------------", len(sel_list))
    matplotlib.rcParams['figure.dpi'] = 500
    plt.figure()
    plt.xlabel('Iterations')
    plt.ylabel('Mutual Information Score')
    # plt.title('Mutual Information Score over iterations')
    plt.plot(np.arange(len(max_info_lst)), max_info_lst,'-o')
    plt.savefig(args.name+'/plots/mutual_info_score.png')
    print(max_info_lst)
    print(sel_list)

    if choose_minimal == True:
        out_index = len(max_info_lst)
        epsilon = 0.05

        for i in range(1, len(max_info_lst)):
            val = max_info_lst[i] - max_info_lst[i-1]
            if val < epsilon:
                out_index = i
                break
        print(out_index)
        sel_list = sel_list[:out_index]
        print(sel_list)
    #exit(0)

    for key in net_dict:
        hw_features_per_device = []
        for j in range(len(sel_list)):
            hw_features_per_device.append(net_dict[key][1][sel_list[j]])
        hw_features_cncat.append(hw_features_per_device)


    #If this is not done separately, the code will break
    for key in net_dict:
        net_dict[key][1] = np.delete(net_dict[key][1], sel_list, axis=0)
        net_dict[key][2] = np.delete(net_dict[key][2], sel_list, axis=0)


    return sel_list, hw_features_cncat



def mutual_info(arr, row_list, nrows, ncols):
    arr_temp = arr[row_list, :]
    t = tuple(arr_temp[i, :] for i in np.arange(len(row_list) - 1, -1, -1))
    inds = np.lexsort(t)
    a_sorted = arr_temp[:, inds]

    mutual_info = 0
    k = 0

    for i in range(1, ncols):
        k+=1
        if not np.array_equal(a_sorted[:,i-1], a_sorted[:,i]):
            mutual_info -= (k/ncols)*np.log(k/ncols)
            k=0

    a_sorted = np.sort(a_sorted[-1, :])
    self_info = 0
    k = 0
    for i in range(1, ncols):
        k += 1
        if not a_sorted[i] == a_sorted[i-1]:
            self_info -= (k/ncols)*np.log(k/ncols)
            k = 0

    # print(row_list[-1], self_info, mutual_info, self_info-mutual_info)

    return self_info - mutual_info

def learn_individual_models(list_val_dict, splitPercentage=0.99, shuffleFeatures=True):
    global maxVal
    ## Identifying the max latency
    for key in list_val_dict:
        maxValTemp = np.amax(list_val_dict[key][1])
        if maxValTemp > maxVal:
            maxVal = maxValTemp

    ##Normalizing the latency by the maxlatency
    for key in list_val_dict:
        list_val_dict[key][1] = list_val_dict[key][1] / maxVal

    for key in list_val_dict:
        if args.model == "lstm":
            learn_lstm_model(key, list_val_dict[key][0], list_val_dict[key][1], list_val_dict[key][2], list_val_dict[key][2].shape[2])
        elif args.model == "xgb":
            # learn_xgb_model_collab(key, list_val_dict[key][0], list_val_dict[key][1], list_val_dict[key][2], list_val_dict[key][2].shape[2], splitPercentage, shuffleFeatures)
            learn_xgb_model(key, list_val_dict[key][0], list_val_dict[key][1], list_val_dict[key][2], list_val_dict[key][2].shape[2], splitPercentage, shuffleFeatures)


'''
Holds out one hardware at a time and learns a combined model for the remaining hardware and tries to
predict for the held-out hardware without any fine-tuning
'''

def learn_collaborative_models(list_val_dict):
    final_indices = 0
    if args.sampling_type == "random":
        final_indices = random_indices(args.numSamples)

    splitVal = 0.2
    ### Take a new set and see how it works
    list_val_dict_set1 = dict(list(list_val_dict.items())[int(0.9*(len(list_val_dict))):])
    list_val_dict_rem  = dict(list(list_val_dict.items())[:int(0.9*(len(list_val_dict)))])

    ## Split the hardware into a smaller and a larger set
    list_val_dict_small = dict(list(list_val_dict_rem.items())[:int(splitVal*(len(list_val_dict_rem)))])
    list_val_dict_large = dict(list(list_val_dict_rem.items())[int(splitVal*(len(list_val_dict_rem))):])


    if args.model == "lstm":
        model, modellist, extractor, final_indices, final_lat, final_features = subsetAndLearn(list_val_dict_small, final_indices, args.numSamples)
    elif args.model == "xgb":
        model, final_indices, final_lat, final_features = subsetAndLearn(list_val_dict_small, final_indices, args.numSamples)


    ####### Test Transfer for each hardware #####################


    ######## Transfer for the remaining held-out set #############

    list_val_dict_set1, hw_features_cncat = cncatHardwareRep(list_val_dict_set1, final_indices)

    final_lat_set1, final_features_set1 = append_with_net_features(list_val_dict_set1, hw_features_cncat)

    if args.model == "lstm":
        checkTransfer(final_lat_set1, final_features_set1, model, final_indices, modellist, extractor, hardware="Set1")
    elif args.model == "xgb":
        checkTransfer(final_lat_set1, final_features_set1, model, final_indices, hardware="Set1")

    ################### Evaluate how many network inputs are required to learn the same model ###########
    #for key in list_val_dict_set1:
    #    for i in range


    ################### Fine tune the model with the new batched hardware set ###########################

    ################### Continue this experiment for more batches! ######################################


def cncatHardwareRep(net_dict, final_indices):
    for key in net_dict:
        net_dict[key][2] = net_dict[key][2][:numLatency,:,:]
        net_dict[key][1] = net_dict[key][1][:numLatency]

    hw_features_cncat = []
    for key in net_dict:
        hw_features_per_device = []
        for j in range(len(final_indices)):
            hw_features_per_device.append(net_dict[key][1][final_indices[j]])
        hw_features_cncat.append(hw_features_per_device)

    #If this is not done separately, the code will break
    for key in net_dict:
        net_dict[key][1] = np.delete(net_dict[key][1], final_indices, axis=0)
        net_dict[key][2] = np.delete(net_dict[key][2], final_indices, axis=0)

    return net_dict, hw_features_cncat


def subsetAndLearn(net_dict, final_indices, numSamples):

    if args.sampling_type == 'random':
        hw_features_cncat = random_sampling(net_dict, final_indices, numSamples)
    elif args.sampling_type == 'statistical':
        final_indices, hw_features_cncat = sample_hwrepresentation(net_dict, numSamples)
    elif args.sampling_type == 'mutual_info':
        final_indices, hw_features_cncat = mutual_information_v2(net_dict, numSamples, choose_minimal=False)
    elif args.sampling_type == 'spearmanCorr':
        final_indices, hw_features_cncat = spearmanCorr(net_dict, numSamples)
    elif args.sampling_type == 'pearsonCorr':
        final_indices, hw_features_cncat = pearsonCorr(net_dict, numSamples)
    else:
        print("Invalid --sampling_type - Fix")
        exit(0)

    dumpSelectedNetworks(final_indices)
    final_lat, final_features = append_with_net_features(net_dict, hw_features_cncat)
    print(final_lat.shape, final_features.shape)
    #final_lat = final_lat / np.amax(final_lat)
    #print(list_val_dict[key][0], final_lat.shape, final_features.shape)
    files = glob.glob('*.txt')
    hardware = 'Mixed Model'
    if args.model=='lstm':
        model, modellist, extractor = learn_lstm_model(hardware, net_dict[files[0]][0], final_lat, final_features, final_features.shape[2])
        return model, modellist, extractor, final_indices, final_lat, final_features
    elif args.model=='xgb':
        model = learn_xgb_model(hardware, net_dict[files[0]][0], final_lat, final_features, final_features.shape[2])
        return model, final_indices, final_lat, final_features


def checkTransfer(lat, features, model, final_indices, modellist = None, extractor = None, hardware="Mixed Model"):

    global maxVal

    testf = features
    testy = lat

    if args.model == 'lstm':
        print(testf.shape, testy.shape)

        testPredict = model.predict(testf)
        testScore = math.sqrt(mean_squared_error(testy, testPredict))
        writeToFile('Transfer Test Score: %f RMSE' % (testScore))
        r2_score = sklearn.metrics.r2_score(testy, testPredict)
        s_coefficient, pvalue = spearmanr(testy, testPredict)
        writeToFile("The transferred R^2 Value for Held out set is: %f"%(r2_score))
        writeToFile("The transferred Spearnman Coefficient and p-value for Held-out set is: %f and %f"%(s_coefficient, pvalue))

        plt.figure()
        plt.xlabel("Actual Latency (in ms)")
        plt.ylabel("Predicted Latency (in ms)")
        sns.scatterplot(testy, testPredict[:,0])
        #plt.title(hold_out_key+'TPear R2:'+str(r2_score)+' TSpear R2:'+str(s_coefficient))
        plt.savefig(args.name+'/plots/'+hardware+'_transferFC.png')

        testPredict = extractor.predict(testf)

        for name, model_lowB in modellist:
            modeltestPred = model_lowB.predict(testPredict)
            testScore = math.sqrt(mean_squared_error(testy, modeltestPred))
            r2_score = sklearn.metrics.r2_score(testy, modeltestPred)
            s_coefficient, pvalue = spearmanr(testy, modeltestPred)
            writeToFile('Transfer Test Score with %s : %f RMSE' % (name, testScore))
            writeToFile("Transfer The R^2 Value with %s for %s: %f"%(hardware, name, r2_score))
            writeToFile("Transfer The Spearnman Coefficient and p-value for %s with %s : %f and %f"%(hardware, name, s_coefficient, pvalue))
            plt.figure()
            plt.xlabel("Actual Latency (in ms)")
            plt.ylabel("Predicted Latency (in ms)")
            sns.scatterplot(testy, modeltestPred)
            #plt.title(name + hardware+' R2: '+str(r2_score)+' SpearVal: '+str(s_coefficient))
            plt.savefig(args.name+'/plots/'+hardware+args.learning_type+'_'+name+'_Transfer.png')

    elif args.model == 'xgb':
        testf  = np.reshape(testf, (testf.shape[0], testf.shape[1]*testf.shape[2]))

        print(testf.shape, testy.shape)
        testPredict = model.predict(testf)
        testScore = math.sqrt(mean_squared_error(testy, testPredict))
        writeToFile('Transfer Test Score: %f RMSE' % (testScore))
        r2_score = sklearn.metrics.r2_score(testy, testPredict)
        s_coefficient, pvalue = spearmanr(testy, testPredict)
        writeToFile("The transferred R^2 Value for Held out set is: %f"%(r2_score))
        writeToFile("The transferred Spearnman Coefficient and p-value for Held-out set is: %f and %f"%(s_coefficient, pvalue))

        testyPlot = testy * maxVal
        testPredictPlot = testPredict * maxVal
        testPlotScore = math.sqrt(mean_squared_error(testyPlot, testPredictPlot))
        writeToFile('Normalized Transfer Test Score: %f RMSE' % (testPlotScore))

        np.savetxt(args.name+'/meta/'+'testy.txt', testyPlot, delimiter='\n')
        np.savetxt(args.name+'/meta/'+'testPredict.txt', testPredictPlot, delimiter='\n')

        plt.figure()
        plt.xlabel("Actual Latency (in ms)")
        plt.ylabel("Predicted Latency (in ms)")
        sns.scatterplot(testy, testPredict, s=15)
        #plt.title(hold_out_key+'TPear R2:'+str(r2_score)+' TSpear R2:'+str(s_coefficient))
        plt.savefig(args.name+'/plots/'+hardware+'_transferFC_scaled_down.png')

        matplotlib.rcParams['figure.dpi'] = 500
        plt.figure()
        plt.xlabel("Actual Latency (in ms)")
        plt.ylabel("Predicted Latency (in ms)")
        sns.scatterplot(testyPlot, testPredictPlot, s=15)
        #plt.title(hold_out_key+'TPear R2:'+str(r2_score)+' TSpear R2:'+str(s_coefficient))
        plt.savefig(args.name+'/plots/'+hardware+'_transferFC_Scaled_up.png')

        matplotlib.rcParams['figure.dpi'] = 500
        plt.figure()
        plt.xlabel("Actual Latency (in ms)")
        plt.ylabel("Predicted Latency (in ms)")
        sns.regplot(x=testyPlot, y=testPredictPlot, scatter_kws={'s':10, 'color':'blue'})
        plt.savefig(args.name+'/plots/'+hardware+'_transferFCregPlot.png')

        RMSEError(testy, testPredict)
        calcErrors(testy, testPredict)


def learn_combined_models(list_val_dict):
    final_indices = 0
    if args.sampling_type == 'random':
        final_indices = random_indices(args.numSamples)

    global maxVal
    ## Identifying the max latency
    for key in list_val_dict:
        maxValTemp = np.amax(list_val_dict[key][1])
        if maxValTemp > maxVal:
            maxVal = maxValTemp

    ##Normalizing the latency by the maxlatency
    for key in list_val_dict:
        list_val_dict[key][1] = list_val_dict[key][1] / maxVal

    ## Splitting the dictionary into 70% and 30%
    list_val_dict_70 = dict(list(list_val_dict.items())[:int(0.7*(len(list_val_dict)))])
    list_val_dict_30 = dict(list(list_val_dict.items())[int(0.7*(len(list_val_dict))):])

    print(len(list_val_dict), len(list_val_dict_70), len(list_val_dict_30))

    if args.model == "lstm":
        model, modellist, extractor, final_indices, final_lat, final_features = subsetAndLearn(list_val_dict_70, final_indices, args.numSamples)
    elif args.model == "xgb":
        model, final_indices, final_lat, final_features = subsetAndLearn(list_val_dict_70, final_indices, args.numSamples)

    ######## Transfer for the remaining held-out set #############

    list_val_dict_30, hw_features_cncat = cncatHardwareRep(list_val_dict_30, final_indices)

    final_lat_30, final_features_30 = append_with_net_features(list_val_dict_30, hw_features_cncat)

    if args.model == "lstm":
        checkTransfer(final_lat_30, final_features_30, model, final_indices, modellist, extractor)
    elif args.model == "xgb":
        checkTransfer(final_lat_30, final_features_30, model, final_indices)

    ########## Calculate Type I and Type II errors ###################

    #calcErrors(testy, testPredict)


from itertools import product
from itertools import combinations
from scipy.spatial import distance

def calcErrors(testy, testPredict):
    global maxVal
    testy = testy * maxVal
    testPredict = testPredict * maxVal

    print(testy.shape, testPredict.shape)

    #print(testy, testPredict)

    ## testy has each hardware's latency stacked up - one after the other - first 118, second 118 and so on
    hardwareRange = int(math.ceil(testy.shape[0] / (numLatency-args.numSamples)))
    print(hardwareRange)

    networkRange = numLatency - args.numSamples

    type1ErrP = []
    type2ErrP = []

    for i in range(hardwareRange):
        testy_hardware = testy[i*networkRange:(i+1)*networkRange]
        testPredict_hardware = testPredict[i*networkRange:(i+1)*networkRange]
        #print(testy_hardware.shape, testPredict_hardware.shape)

        c = list(combinations(testy_hardware,2))
        d = list(combinations(testPredict_hardware, 2))
        #c = list(product(testy_hardware, testy_hardware))
        #d = list(product(testPredict_hardware, testPredict_hardware))
        #print(len(c), len(d))

        #print("================================ Hardware %d =========================="%(i))

        typeThres = 0.1

        distance_testy = np.ones(len(c))
        distance_testPredict = np.ones(len(d))

        for j in range(distance_testy.shape[0]):
            distance_testy[j] = distance.euclidean(c[j][0], c[j][1])
            distance_testPredict[j] = distance.euclidean(d[j][0], d[j][1])

        #print(distance_testy.shape, distance_testPredict.shape)

        type1Err = 0
        type2Err = 0

        for j in range(distance_testy.shape[0]):
            dev1 = (distance_testy[j] - distance_testPredict[j]) / distance_testy[j]
            dev2 = (distance_testPredict[j] - distance_testy[j]) / distance_testPredict[j]
            if (distance_testy[j] > distance_testPredict[j]) and (dev1 > typeThres):
                type1Err += 1
            elif (distance_testPredict[j] > distance_testy[j]) and (dev2 > typeThres):
                type2Err +=1
        #print("For Hardware %d - Type1Err Percentage: %f, Type2Err Percentage: %f, Threshold: %f"%(i,(type1Err/distance_testy.shape[0])*100,(type2Err/distance_testy.shape[0])*100, typeThres))
        type1ErrP.append((type1Err/distance_testy.shape[0])*100)
        type2ErrP.append((type2Err/distance_testy.shape[0])*100)

    print(mean(type1ErrP), mean(type2ErrP))
    writeToFile('Type-1 Error: ' +str(mean(type1ErrP)) + '  Type-2 Error: ' +str(mean(type2ErrP)))
    writeToFileError(type1ErrP, type2ErrP)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def RMSEError(testy, testPredict):
    global maxVal
    testy = testy * maxVal
    testPredict = testPredict * maxVal


    ## testy has each hardware's latency stacked up - one after the other - first 118, second 118 and so on
    networkRange = numLatency - args.numSamples
    hardwareRange = int(math.ceil(testy.shape[0] / networkRange))
    print(hardwareRange)

    l = []
    for i in range(hardwareRange):
        testy_hardware = testy[i*networkRange:(i+1)*networkRange]
        testPredict_hardware = testPredict[i*networkRange:(i+1)*networkRange]
        r2_score = sklearn.metrics.r2_score(testy_hardware, testPredict_hardware)
        rmse = math.sqrt(mean_squared_error(testy_hardware, testPredict_hardware))
        mape = mean_absolute_percentage_error(testy_hardware, testPredict_hardware)
        avg = np.mean(testy_hardware)
        med = np.median(testy_hardware)
        l.append([r2_score, rmse, np.amin(testy_hardware), np.amax(testy_hardware), avg, med, mape])
    dumpRMSE(l)

def dumpRMSE(l):
    meta = open(args.name+'/meta/RMSE.txt', "w")
    for i in l:
        s = ','.join(map(str, i))
        meta.write(s+'\n')
    meta.close()

def writeToFileError(l1, l2):
    meta = open(args.name+'/meta/error.txt', "w")
    for i in range(len(l1)):
        s = str(l1[i]) + ',' + str(l2[i]) + ',' + '\n'
        meta.write(s)
    meta.close()

def writeToFile(stringVal):
    meta = open(args.name+'/meta/metadata.txt', "a")
    meta.write(stringVal)
    meta.write('\n')
    meta.close()
    print(stringVal)

def dumpSelectedNetworks(s):
    file = open(args.name+'/meta/networkindices.txt', "w")
    text = ''
    for i in range(len(s)):
        text = text + str(s[i]) + ','
    file.write(text)
    file.write('\n')
    file.close()
    print(s)


def main():
    list_val_dict = {}
    features, maxLayers = parse_features()
    files = glob.glob('mobileData/*.txt')
    for file in files:
        latency = parse_latency(file)
        tmp_list = []
        tmp_list.append(maxLayers)
        tmp_list.append(latency)
        tmp_list.append(features)
        list_val_dict[file] = tmp_list
    if args.model != 'lstm' and args.model != 'xgb':
        print("Invalid--model")
        exit(0)
    if args.learning_type == 'individual':
        learn_individual_models(list_val_dict)
    elif args.learning_type == 'combined':
        learn_combined_models(list_val_dict)
    elif args.learning_type == 'collaborative':
        learn_collaborative_models(list_val_dict)
    else:
        print("Invalid --learning_type - Fix")
        exit(0)

    # learn_combined_models(list_val_dict)
    #learn_individual_models(list_val_dict)
    # plotLatnecyRandomSamples(list_val_dict)
    # plotLatnecyStatSamples(list_val_dict)
    # plotLatnecyMISamples(list_val_dict)

if __name__ == '__main__':
    sns.set()
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    parser = argparse.ArgumentParser(description = "Generalizable Cost Models")
    parser.add_argument("--sampling_type", type = str, help = 'Enter the Sampling Type to be used on the data. Options are random, mutual_info, spearmanCorr, statistical', required=True)
    parser.add_argument("--learning_type", type = str, help = 'Enter the Learning Type to be used on the data. Options are individual, combined, collaborative', required=True)
    parser.add_argument("--name", type=str, help = 'Name of the run', required=True)
    parser.add_argument("--numSamples", type=int, help = 'Number of Benchmark Samples', required=True)
    parser.add_argument("--model", type=str, help='Model to be trained', required=True)
    args = parser.parse_args()
    os.mkdir(args.name)
    os.mkdir(args.name+'/models')
    os.mkdir(args.name+'/plots')
    os.mkdir(args.name+'/meta')
    main()



