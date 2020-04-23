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

def parse_features(subdir, latency_file, embeddings):
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
      for j in range(len(temp)):
            maxFlops=max(maxFlops, int(temp[j][12]))
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

def learn_lstm_model(hardware, maxLayer, lat_mean, features, featuresShape):
  
  features, lat_mean = shuffle(features,lat_mean)
  trainf = features[:int(0.85*len(features))]
  trainy = lat_mean[:int(0.85*len(features))]
  testf = features[int(0.85*len(features)):]
  testy = lat_mean[int(0.85*len(features)):]
  print(trainf.shape, trainy.shape, testf.shape, testy.shape)
  
  #Create an LSTM model
  model=Sequential()
  model.add(Masking(mask_value=-1,input_shape=(maxLayer, featuresShape)))
  model.add(LSTM(20, activation='relu'))
  model.add(Dense(1))
  '''Adam intialized with Default Values. Tune only intial Learning rate.
  opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)'''
  opt = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
  model.compile(loss='mean_squared_error', optimizer=opt)
  model.summary()
  model.fit(trainf, trainy, epochs=800, batch_size=2048, verbose=1)

  trainPredict = model.predict(trainf)
  testPredict = model.predict(testf)
  trainScore = math.sqrt(mean_squared_error(trainy, trainPredict))
  print('Train Score: %f RMSE' % (trainScore))
  testScore = math.sqrt(mean_squared_error(testy, testPredict))
  r2_score = sklearn.metrics.r2_score(testy, testPredict)
  s_coefficient, pvalue = spearmanr(testy, testPredict)
  print('Test Score: %f RMSE' % (testScore))
  print("The R^2 Value for %s:"%(hardware), r2_score)
  print("The Spearnman Coefficient and p-value for %s: %f and %f"%(hardware), s_coefficient, pvalue)
  
  plt.figure()
  plt.xlabel("Actual Latency")
  plt.ylabel("Predicted Latency")
  plt.scatter(testy, testPredict[:,0])
  plt.title(hardware+' R2: '+str(r2_score))
  plt.savefig(hardware+'.png')
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
        net_dict[key][2] = net_dict[key][2][:5000,:,:] #Not required actually.. Simply doing
        net_dict[key][1] = net_dict[key][1][:5000]
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
        rand_indices.append(random.randint(0,5000))
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
        net_dict[key][2] = net_dict[key][2][:5000,:,:]
        net_dict[key][1] = net_dict[key][1][:5000]

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
        net_dict[key][2] = net_dict[key][2][:5000,:,:]
        net_dict[key][1] = net_dict[key][1][:5000]

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

### Prof. Pratyush's MI implementation

def mutual_information_v2(net_dict, numSamples):
    index = 0
    ## Rows - Networks, Columns - Hardware

    for key in net_dict:
        net_dict[key][2] = net_dict[key][2][:5000,:,:]
        net_dict[key][1] = net_dict[key][1][:5000]

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
    sel_list = [val]
    hw_features_cncat = []

    print( " ------------------------------------- Beginning Sampling -------------------")
    for k in range(numSamples-1):
        max_info = 0
        for i in range(nrows):
            if i in sel_list:
                continue
            m = -mutual_info(stacked_arr, sel_list + [i], nrows, ncols)

            if m >= max_info:
                max_index = i
                max_info = m
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

    return sel_list, hw_features_cncat



def mutual_info(arr, row_list, nrows, ncols):
    arr_temp = arr[row_list, :]
    t = tuple(arr_temp[i, :] for i in np.arange(len(row_list) - 1, -1, -1))
    inds = np.lexsort(t)
    a_sorted = arr_temp[:, inds]

    self_info = 0
    k = 0

    for i in range(1, ncols):
        k+=1
        if not np.array_equal(a_sorted[:,i-1], a_sorted[:,i]):
            self_info += k*np.log(k)
            k=0

    a_sorted = a_sorted[-1, :]
    mutual_info = 0
    k = 0
    for i in range(1, ncols):
        k += 1
        if not a_sorted[i] == a_sorted[i-1]:
            mutual_info += k * np.log(k)
            k = 0

    return self_info - mutual_info



def learn_individual_models(list_val_dict):
    for key in list_val_dict:
      learn_lstm_model(key, list_val_dict[key][0], list_val_dict[key][1], list_val_dict[key][2], 13)


'''
Holds out one hardware at a time and learns a combined model for the remaining hardware and tries to
predict for the held-out hardware without any fine-tuning
'''
def learn_combined_models(list_val_dict):
    #maxSamples = 30
    #final_indices = random_indices(maxSamples)
    for key in list_val_dict:
        #list_val_dict_local = list_val_dict.copy() #This was creating a shallow copy
        list_val_dict_local = copy.deepcopy(list_val_dict)
        hold_out_val = list_val_dict_local[key]
        hold_out_key = key
        print("-------------------Check-------------------: %n ",list_val_dict_local[key][2].shape, list_val_dict[key][2].shape, hold_out_val[2].shape)
        list_val_dict_local.pop(key)
        print("%n", len(list_val_dict_local), len(list_val_dict), key)
        
        #hw_features_cncat = random_sampling(list_val_dict_local, final_indices, maxSamples)
        # final_indices, hw_features_cncat = sample_hwrepresentation(list_val_dict_local, 30)
        # final_indices, hw_features_cncat = mutual_information(list_val_dict_local, 30)
        final_indices, hw_features_cncat = mutual_information_v2(list_val_dict_local, 30)

        final_lat, final_features = append_with_net_features(list_val_dict_local, hw_features_cncat)
        #print(list_val_dict[key][0], final_lat.shape, final_features.shape)
        model = learn_lstm_model('Mixed Without'+hold_out_key, list_val_dict[key][0], final_lat, final_features, final_features.shape[2])

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
        print("The Spearnman Coefficient and p-value for %s: %f and %f"%(hardware), s_coefficient, pvalue)

        plt.figure()
        plt.xlabel("Transfer : Actual Latency")
        plt.ylabel("Transfer : Predicted Latency")
        plt.scatter(testy, testPredict[:,0])
        plt.title(hold_out_key+'Transfer R2:'+str(r2_score))
        plt.savefig(hold_out_key+'transfer'+'.png')

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
    fig.savefig("plots/RandomSampling.png")

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
        fig.savefig("plots/Statistical_Without_"+str(hold_out_key)+".png")

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
        fig.savefig("plots/MutualInfo_Without_"+str(hold_out_key)+".png")

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

    # learn_combined_models(list_val_dict)
    # learn_individual_models(list_val_dict)
    plotLatnecyRandomSamples(list_val_dict)
    plotLatnecyStatSamples(list_val_dict)
    plotLatnecyMISamples(list_val_dict)

if __name__ == '__main__':
    np.random.seed(42)
    # tf.random.set_seed(42)
    random.seed(30)
    main()



