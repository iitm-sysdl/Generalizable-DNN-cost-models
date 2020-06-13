import keras
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
#from tensorflow.keras import layers
#from tensorflow.keras import optimizers
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Masking
from keras.layers import Input
from keras.layers import Concatenate
from keras import optimizers
from scipy.stats import spearmanr
from scipy import stats
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
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
numLatency = 100
lat = []
def parse_latency(file):
    global lat
    data = np.genfromtxt(file, delimiter=',')
    latency = np.mean(data, axis=1)
    latency = latency[:100]
    lat.append(latency)
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
  trainf = features[:int(0.7*len(features))]
  trainy = lat_mean[:int(0.7*len(features))]
  #testf = features[:int(1.0*len(features))]
  #testy = lat_mean[:int(1.0*len(features))]
  testf = features[int(0.7*len(features)):]
  testy = lat_mean[int(0.7*len(features)):]
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
  filepath='model.hdf5'
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')#montor can be val_loss or loss
  es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=50)
  val = model.fit(trainf, trainy, epochs=250, batch_size=512, verbose=1, callbacks=[es, checkpoint], validation_data=(testf, testy))
  model.load_weights(filepath)
#   mlflow.set_tag('Optim', opt)
#   mlflow.set_tag('sampling_type', args.sampling_type)
#   mlflow.set_tag('learning_type', args.learning_type)


  #with mlflow.start_run() as run:
  #  mlflow.log_keras_model(model, "runs")

  #mlflow.keras.save_model(model, "model")

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
  sns.scatterplot(testy, testPredict[:,0])
  #plt.title(hardware+' R2: '+str(r2_score)+' SpearVal: '+str(s_coefficient))
  plt.savefig(hardware+"_"+args.learning_type+'.png')

  ind = np.argsort(testy)

  testy = testy[ind]
  testf = testf[ind]

  #testf = np.sort(testf)

  testf_25 = testf[:int(0.25*len(testf))]
  testy_25 = testy[:int(0.25*len(testy))]
  testPredict = model.predict(testf_25)
  testScore = math.sqrt(mean_squared_error(testy_25, testPredict))

  r2_score = sklearn.metrics.r2_score(testy_25, testPredict)
  s_coefficient, pvalue = spearmanr(testy_25, testPredict)

  print('Test Score: %f RMSE' % (testScore))
  print("The R^2 Value for %s:"%(hardware), r2_score)
  print("The Spearnman Coefficient and p-value for %s: %f and %f"%(hardware, s_coefficient, pvalue))


  plt.figure()
  plt.xlabel("Actual Latency")
  plt.ylabel("Predicted Latency")
  sns.scatterplot(testy_25, testPredict[:,0])
  #plt.title(hardware+' F25% R2: '+str(r2_score)+' SpearVal: '+str(s_coefficient))
  plt.savefig(hardware+"_"+args.learning_type+'_25percentile.png')


  testf_50 = testf[int(0.25*len(testf)):int(0.50*len(testf))]
  testy_50 = testy[int(0.25*len(testy)):int(0.50*len(testy))]
  testPredict = model.predict(testf_50)
  testScore = math.sqrt(mean_squared_error(testy_50, testPredict))

  r2_score = sklearn.metrics.r2_score(testy_50, testPredict)
  s_coefficient, pvalue = spearmanr(testy_50, testPredict)

  print('Test Score: %f RMSE' % (testScore))
  print("The R^2 Value for %s:"%(hardware), r2_score)
  print("The Spearnman Coefficient and p-value for %s: %f and %f"%(hardware, s_coefficient, pvalue))

  plt.figure()
  plt.xlabel("Actual Latency")
  plt.ylabel("Predicted Latency")
  sns.scatterplot(testy_50, testPredict[:,0])
  #plt.title(hardware+' S25% R2: '+str(r2_score)+' SpearVal: '+str(s_coefficient))
  plt.savefig(hardware+"_"+args.learning_type+'_50percentile.png')


  testf_75 = testf[int(0.50*len(testf)):int(0.75*len(testf))]
  testy_75 = testy[int(0.50*len(testy)):int(0.75*len(testy))]
  testPredict = model.predict(testf_75)
  testScore = math.sqrt(mean_squared_error(testy_75, testPredict))

  r2_score = sklearn.metrics.r2_score(testy_75, testPredict)
  s_coefficient, pvalue = spearmanr(testy_75, testPredict)

  print('Test Score: %f RMSE' % (testScore))
  print("The R^2 Value for %s:"%(hardware), r2_score)
  print("The Spearnman Coefficient and p-value for %s: %f and %f"%(hardware, s_coefficient, pvalue))

  plt.figure()
  plt.xlabel("Actual Latency")
  plt.ylabel("Predicted Latency")
  sns.scatterplot(testy_75, testPredict[:,0])
  #plt.title(hardware+' T25% R2: '+str(r2_score)+' SpearVal: '+str(s_coefficient))
  plt.savefig(hardware+"_"+args.learning_type+'_75percentile.png')


  testf_100 = testf[int(0.75*len(testf)):]
  testy_100 = testy[int(0.75*len(testy)):]
  testPredict = model.predict(testf_100)
  testScore = math.sqrt(mean_squared_error(testy_100, testPredict))

  r2_score = sklearn.metrics.r2_score(testy_100, testPredict)
  s_coefficient, pvalue = spearmanr(testy_100, testPredict)

  print('Test Score: %f RMSE' % (testScore))
  print("The R^2 Value for %s:"%(hardware), r2_score)
  print("The Spearnman Coefficient and p-value for %s: %f and %f"%(hardware, s_coefficient, pvalue))

  plt.figure()
  plt.xlabel("Actual Latency")
  plt.ylabel("Predicted Latency")
  sns.scatterplot(testy_100, testPredict[:,0])
  #plt.title(hardware+' Fo25% R2: '+str(r2_score)+' SpearVal: '+str(s_coefficient))
  plt.savefig(hardware+"_"+args.learning_type+'_100percentile.png')

  extractor = Model(outputs=model.get_layer('fc').input, inputs=model.input)
  extractor.summary()
  from sklearn.neighbors import KNeighborsRegressor
  for i in [1,2,3,4,5,6]:
    for j in ['uniform', 'distance']:
        for k in ['auto', 'ball_tree', 'kd_tree', 'brute']:
            with mlflow.start_run():
                mlflow.log_param('Num Neighbours', str(i))
                mlflow.log_param('Weights', str(j))
                mlflow.log_param('Algorithm', str(k))
                knn = KNeighborsRegressor(n_neighbors=i, weights=j, algorithm=k)
                trainPredict = extractor.predict(trainf)
                testPredict = extractor.predict(testf)
                knn.fit(trainPredict, trainy)
                knntestPred = knn.predict(testPredict)
                testScore = math.sqrt(mean_squared_error(testy, knntestPred))
                r2_score = sklearn.metrics.r2_score(testy, knntestPred)
                s_coefficient, pvalue = spearmanr(testy, knntestPred)
                print('Test Score with kNN : %f RMSE' % (testScore))
                print("The R^2 Value with kNN for %s:"%(hardware), r2_score)
                print("The Spearnman Coefficient and p-value for %s with kNN : %f and %f"%(hardware, s_coefficient, pvalue))
                plt.figure()
                plt.xlabel("Actual Latency")
                plt.ylabel("Predicted Latency")
                sns.scatterplot(testy, knntestPred)
                plt.title('kNN' + hardware+' R2: '+str(r2_score)+' SpearVal: '+str(s_coefficient))
                plt.savefig(hardware+args.learning_type+'_knn-'+str(i)+'-'+j+'-'+k+'.png')
                mlflow.log_metric('R Square',r2_score)
                mlflow.log_metric('Spearman',s_coefficient)
  #plt.show()
  return model






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
        rand_indices.append(random.randint(0,numLatency))
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

'''
Computes Pairwise Mutual Information
'''

def parallel_mi(i, matShape, stacked_arr, mutualInformationMatrix):
    for j in range(i, matShape):
            mutualInformationMatrix[i][j] = mutualInformationMatrix[j][i] = sklearn.metrics.normalized_mutual_info_score(stacked_arr[i], stacked_arr[j])

def mutual_information(net_dict, numSamples):
    index = 0

    for key in net_dict:
        net_dict[key][2] = net_dict[key][2][:numLatency,:,:]
        net_dict[key][1] = net_dict[key][1][:numLatency]

    for key in net_dict:
        if index == 0:
            stacked_arr = net_dict[key][1]
        else:
            stacked_arr = np.column_stack((stacked_arr, net_dict[key][1]))
        index+=1
    matShape = stacked_arr.shape[0]
    print(stacked_arr.shape)
    mutualInformationMatrix = np.zeros((matShape,matShape))

    processes = []
    print("-------------------------------------------Begin PreComputation----------------------------------------------------")
    for i in range(matShape//100):
        for j in range(100):
            p = mp.Process(target=parallel_mi, args=(i*100+j, matShape, stacked_arr, mutualInformationMatrix))
            processes.append(p)
            p.start()

        for process in processes:
            process.join()
        print("%i Done" %((i+1)*100))
    print("-------------------------------------------Done PreComputation----------------------------------------------------")
    val = np.random.randint(0, stacked_arr.shape[0])
    sel_list = [val]
    hw_features_cncat = []

    print( " ------------------------------------- Beginning Sampling -------------------")
    for k in range(numSamples-1):
        mininfo=1000000000000
        for l in range(stacked_arr.shape[0]):
            if l in sel_list:
                continue
            temp = mutualInformationMatrix[l][sel_list].sum()
            if temp < mininfo:
                mininfo=temp
                min_index = l
        sel_list = sel_list + [l]

    print(" ------------------------------- Done Sampling -----------------------------", len(sel_list))
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

def corr_choose(rho, maxSamples, threshold = 0.97, stop_condition = 5, debug=True):
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
        if debug:
            print('Iteration', i, ": Number of remaining vectors", rho.shape[0])
        if len(indices) <= stop_condition:
            break
    if debug:
        print('Chosen networks are ', subset)
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

    sel_list = corr_choose(rho, 10, 0.98)
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

    sel_list = corr_choose(rho, 10, 0.98)
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



### Prof. Pratyush's MI implementation

def mutual_information_v2(net_dict, numSamples):
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
    val = np.random.randint(0, nrows)
    #val = select_network()
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
    for key in net_dict:
        hw_features_per_device = []
        for j in range(len(sel_list)):
            hw_features_per_device.append(net_dict[key][1][sel_list[j]])
        hw_features_cncat.append(hw_features_per_device)

    #If this is not done separately, the code will break
    for key in net_dict:
        net_dict[key][1] = np.delete(net_dict[key][1], sel_list, axis=0)
        net_dict[key][2] = np.delete(net_dict[key][2], sel_list, axis=0)

    plt.figure()
    plt.xlabel('Iterations')
    plt.ylabel('Mutual Information Score')
    plt.title('Mutual Information Score over iterations')
    plt.plot(np.arange(len(max_info_lst)), max_info_lst,'-o')
    plt.savefig('mutual_info_score.png')
    print(max_info_lst)
    print(sel_list)

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

def learn_individual_models(list_val_dict):
    for key in list_val_dict:
      learn_lstm_model(key, list_val_dict[key][0], list_val_dict[key][1], list_val_dict[key][2], 13)


'''
Holds out one hardware at a time and learns a combined model for the remaining hardware and tries to
predict for the held-out hardware without any fine-tuning
'''
def learn_combined_models(list_val_dict):
    if args.sampling_type == 'random':
        maxSamples = 5
        final_indices = random_indices(maxSamples)
    #for key in list_val_dict:
    #list_val_dict_local = list_val_dict.copy() #This was creating a shallow copy
    #list_val_dict_local = copy.deepcopy(list_val_dict)
    #hold_out_val = list_val_dict_local[key]
    #hold_out_key = key
    #print("-------------------Check-------------------: %n ",list_val_dict_local[key][2].shape, list_val_dict[key][2].shape, hold_out_val[2].shape)
    #list_val_dict_local.pop(key)
    #print("%n", len(list_val_dict_local), len(list_val_dict), key)

    if args.sampling_type == 'random':
        hw_features_cncat = random_sampling(list_val_dict, final_indices, maxSamples)
    elif args.sampling_type == 'statistical':
        final_indices, hw_features_cncat = sample_hwrepresentation(list_val_dict, 30)
    elif args.sampling_type == 'mutual_info_v1':
        final_indices, hw_features_cncat = mutual_information(list_val_dict, 30)
    elif args.sampling_type == 'mutual_info_v2':
        final_indices, hw_features_cncat = mutual_information_v2(list_val_dict, 30)
    elif args.sampling_type == 'spearmanCorr':
        final_indices, hw_features_cncat = spearmanCorr(list_val_dict, 30)
    elif args.sampling_type == 'pearsonCorr':
        final_indices, hw_features_cncat = pearsonCorr(list_val_dict, 30)
    else:
        print("Invalid --sampling_type - Fix")
        exit(0)

    final_lat, final_features = append_with_net_features(list_val_dict, hw_features_cncat)
    #final_lat = final_lat / np.amax(final_lat)
    #print(list_val_dict[key][0], final_lat.shape, final_features.shape)
    files = glob.glob('*.txt')
    model = learn_lstm_model('Mixed Model', list_val_dict[files[0]][0], final_lat, final_features, final_features.shape[2])




    '''
    held_out_hw_feature = []

    #Create a hardware representation for the held-out hardware -- should reuse previous code
    for i in range(len(final_indices)):
        held_out_hw_feature.append(hold_out_val[1][final_indices[i]])
    hold_out_val[1] = np.delete(hold_out_val[1], final_indices, axis=0)
    hold_out_val[2] = np.delete(hold_out_val[2], final_indices, axis=0)

    new_lat_ft = np.tile(held_out_hw_feature, (hold_out_val[2].shape[0], hold_out_val[2].shape[1], 1))
    appended_features = np.concatenate((hold_out_val[2], new_lat_ft), axis=2)

    features, lat = shuffle(appended_features, hold_out_val[1])
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
    print("The transferred R^2 Value for %s:"%(hold_out_key), r2_score)
    print("The transferred Spearnman Coefficient and p-value for %s: %f and %f"%(hold_out_key, s_coefficient, pvalue))

    plt.figure()
    plt.xlabel("Transfer : Actual Latency")
    plt.ylabel("Transfer : Predicted Latency")
    plt.scatter(testy, testPredict[:,0])
    plt.title(hold_out_key+'TPear R2:'+str(r2_score)+' TSpear R2:'+str(s_coefficient))
    plt.savefig(hold_out_key+'transfer'+'.png')
    '''

def main():
    list_val_dict = {}
    features, maxLayers = parse_features()
    files = glob.glob('*.txt')
    for file in files:
        latency = parse_latency(file)
        tmp_list = []
        tmp_list.append(maxLayers)
        tmp_list.append(latency)
        tmp_list.append(features)
        list_val_dict[file] = tmp_list
    if args.learning_type == 'individual':
        learn_individual_models(list_val_dict)
    elif args.learning_type == 'combined':
        learn_combined_models(list_val_dict)
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
    parser = argparse.ArgumentParser(description = "LSTM Models for Transferrable Cost Models")
    parser.add_argument("--sampling_type", type = str, help = 'Enter the Sampling Type to be used on the data. Options are individual, combined', required=True)
    parser.add_argument("--learning_type", type = str, help = 'Enter the Learning Type to be used on the data. Options are random, statistical, mutual_info_v1, mutual_info_v2, spearmanCorr', required=True)
    args = parser.parse_args()
    main()



