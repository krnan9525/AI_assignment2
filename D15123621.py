# D15123621 Nan Yang
# AI assignment2

import numpy as np
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet

from pandas import DataFrame, read_csv

# General syntax to import a library but no functions:
##import (library) as (give the library a nickname/alias)
import matplotlib.pyplot as plt
import pandas as pd  # this is how I usually import pandas
import sys  # only needed to determine Python version number
import matplotlib  # only needed to determine Matplotlib version number

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)

training = pd.read_csv("./data/trainingset.txt", header=None)
queries = pd.read_csv("./data/queries.txt", header=None)

distinguish_arr = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]  # 1=useful 0=useless 2=output
type_arr = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0]  # 1=number 0=text
ds = SupervisedDataSet(16, 1)

# df_norm = (training - training.mean()) / (training.max() - training.min())

# i=0
for column in training:
    temp_arr = []
    arr_output = []
    if(type_arr[column]==1):
        print(column)
        training[column] = (training[column] - training[column].mean()) / (training[column].max() - training[column].min())
    else:
        continue
        training[column] = (training[column] - training[column].mean()) / (training[column].max() - training[column].min())
    for i in range(0, distinguish_arr.__len__()):
        if (distinguish_arr[i] == 1):
            if (isinstance(training[column][i], float)):
                temp_arr.append(training[column][i])
            # if(isinstance(training[column][i], np.basestring)):
                # temp_arr.append(training[column][i].)
        if (distinguish_arr[i] == 2):
            arr_output = [training[column][i]]
    ds.addSample(temp_arr, arr_output)

net = buildNetwork(16, 20, 1)

# print(net.activate([2, 1]))
