import numpy as np
import seaborn as sns 
import pandas as pd
import os
import matplotlib
from matplotlib import pyplot as plt
import sklearn
from sklearn.cluster import KMeans
import fnmatch
import glob
import random
random.seed(42)
np.random.seed(42)
files = glob.glob('*.txt')
print(files) 
print(len(files))   
sns.set()
meanlat = []
pattern = "*flops.txt*"
for file in files:
  if not fnmatch.fnmatch(file, pattern):
    data = pd.read_csv(file, sep=",", header=None)
    data['mean'] = data.mean(axis=1)
    data = data[data.index < 118]
    laten = data['mean'].tolist()
    meanlat.append(laten)
meanlat = np.array(meanlat)
print(meanlat.shape, len(files))
print(meanlat)

print("Hardware Cluster")
model = KMeans(n_clusters=3)
model.fit_transform(meanlat)

i1 = np.where(model.labels_ == 0)
i2 = np.where(model.labels_ == 1)
i3 = np.where(model.labels_ == 2)
print("-----------------Cluster1--------------")
print(list(i1[0]), len(i1[0]))
print("-----------------Cluster2--------------")
print(list(i2[0]), len(i2[0]))
print("-----------------Cluster3--------------")
print(list(i3[0]), len(i3[0]))

print("Network Cluster")
model = KMeans(n_clusters=3)
model.fit_transform(meanlat.T)

i1 = np.where(model.labels_ == 0)
i2 = np.where(model.labels_ == 1)
i3 = np.where(model.labels_ == 2)
print("-----------------Cluster1--------------")
print(list(i1[0]),  len(i1[0]))
print("-----------------Cluster2--------------")
print(list(i2[0]),  len(i2[0]))
print("-----------------Cluster3--------------")
print(list(i3[0]),  len(i3[0]))