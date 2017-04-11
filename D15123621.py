# D15123621 Nan Yang
# AI assignment2

import numpy as np
import pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer
from pandas import DataFrame, read_csv

# General syntax to import a library but no functions:
##import (library) as (give the library a nickname/alias)
import matplotlib.pyplot as plt
import pandas as pd  # this is how I usually import pandas
import sys  # only needed to determine Python version number
import matplotlib  # only needed to determine Matplotlib version number
from pybrain.supervised.trainers import BackpropTrainer

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)
print('Matplotlib version ' + matplotlib.__version__)

with open("./data/trainingset.txt", "rt") as fin:
    with open("./data/trainingset_parsed.txt", "wt") as fout:
        for line in fin:
            line = line.replace('\"JobCat', '\"')
            line = line.replace('\"unknown\"', '\"0\"')
            line = line.replace('\"?\"', '\"0\"')
            line = line.replace('\"yes\"', '\"1\"')
            line = line.replace('\"no\"', '\"0\"')
            line = line.replace('\"married\"', '\"0\"')
            line = line.replace('\"single\"', '\"1\"')
            line = line.replace('\"divorced\"', '\"2\"')
            line = line.replace('\"secondary\"', '\"2\"')
            line = line.replace('\"primary\"', '\"1\"')
            line = line.replace('\"tertiary\"', '\"0\"')
            line = line.replace('\"telephone\"', '\"1\"')
            line = line.replace('\"cellular\"', '\"2\"')
            line = line.replace('\"jan\"', '\"1\"')
            line = line.replace('\"feb\"', '\"2\"')
            line = line.replace('\"mar\"', '\"3\"')
            line = line.replace('\"apr\"', '\"4\"')
            line = line.replace('\"may\"', '\"5\"')
            line = line.replace('\"jun\"', '\"6\"')
            line = line.replace('\"jul\"', '\"7\"')
            line = line.replace('\"aug\"', '\"8\"')
            line = line.replace('\"sep\"', '\"9\"')
            line = line.replace('\"oct\"', '\"10\"')
            line = line.replace('\"nov\"', '\"11\"')
            line = line.replace('\"dec\"', '\"12\"')
            line = line.replace('\"failure\"', '\"0\"')
            line = line.replace('\"other\"', '\"1\"')
            line = line.replace('\"success\"', '\"2\"')
            line = line.replace('\"TypeA\"', '\"0\"')
            line = line.replace('\"TypeB\"', '\"1\"')
            fout.write(line)
fin.close()
fout.close()
training = pd.read_csv("./data/trainingset_parsed.txt", header=None)
distinguish_arr = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]  # 1=useful 0=useless 2=output
type_arr = [2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0]  # 1=number 0=text
ds = SupervisedDataSet(16, 1)
max_arr = training.max()
print(max_arr)
min_arr = training.min()
mean_arr = training.mean()
# df_norm = (training - training.mean()) / (training.max() - training.min())

# i=0
for column in training:
    if(type_arr[column]==1 or type_arr[column]==0):
        #print(column)
        if((training[column].max() - training[column].min() != 0)):
            training[column] = (training[column] - min_arr[column]) / (max_arr[column] - min_arr[column])
        else:
            training[column] = 0.0
    else:
        continue
        training[column] = (training[column] - training[column].mean()) / (training[column].max() - training[column].min())

# print(training)

for index, row in training.iterrows():
    temp_arr = []
    arr_output = []
    #print(index)
    for i in range(0, distinguish_arr.__len__()):
        if (distinguish_arr[i] == 1):
            if (isinstance(row[i], float)):
                temp_arr.append(row[i])
            else:
                continue
            # if(isinstance(training[column][i], np.basestring)):
                # temp_arr.append(training[column][i].)
        if (distinguish_arr[i] == 2):
            arr_output.append(row[i])
    # print(temp_arr)
    ds.addSample(temp_arr, arr_output)

print("loaded finished")

net = buildNetwork(16, 2000, 1)
trainer = BackpropTrainer(net, ds)
trainer.trainEpochs(6)

print(len(ds))


with open("./data/new_queries.txt", "rt") as fin:
    with open("./data/new_queries_parsed.txt", "wt") as fout:
        for line in fin:
            line = line.replace('\"JobCat', '\"')
            line = line.replace('\"unknown\"', '\"0\"')
            line = line.replace('\"?\"', '\"0\"')
            line = line.replace('\"yes\"', '\"1\"')
            line = line.replace('\"no\"', '\"0\"')
            line = line.replace('\"married\"', '\"0\"')
            line = line.replace('\"single\"', '\"1\"')
            line = line.replace('\"divorced\"', '\"2\"')
            line = line.replace('\"secondary\"', '\"2\"')
            line = line.replace('\"primary\"', '\"1\"')
            line = line.replace('\"tertiary\"', '\"0\"')
            line = line.replace('\"telephone\"', '\"1\"')
            line = line.replace('\"cellular\"', '\"2\"')
            line = line.replace('\"jan\"', '\"1\"')
            line = line.replace('\"feb\"', '\"2\"')
            line = line.replace('\"mar\"', '\"3\"')
            line = line.replace('\"apr\"', '\"4\"')
            line = line.replace('\"may\"', '\"5\"')
            line = line.replace('\"jun\"', '\"6\"')
            line = line.replace('\"jul\"', '\"7\"')
            line = line.replace('\"aug\"', '\"8\"')
            line = line.replace('\"sep\"', '\"9\"')
            line = line.replace('\"oct\"', '\"10\"')
            line = line.replace('\"nov\"', '\"11\"')
            line = line.replace('\"dec\"', '\"12\"')
            line = line.replace('\"failure\"', '\"0\"')
            line = line.replace('\"other\"', '\"1\"')
            line = line.replace('\"success\"', '\"2\"')
            line = line.replace('\"TypeA\"', '\"0\"')
            line = line.replace('\"TypeB\"', '\"1\"')
            fout.write(line)
fin.close()
fout.close()

queries = pd.read_csv("./data/new_queries_parsed.txt", header=None)


for column in queries:
    if(type_arr[column]==1 or type_arr[column]==0):
        if((training[column].max() - training[column].min() != 0)):
            queries[column] = (queries[column] - min_arr[column]) / (max_arr[column]-min_arr[column])
        else:
            queries[column] = 0.0
right_num = 0
wrong_num = 0
for index2, row in queries.iterrows():
    temp_arr = []
    for i in range(0,distinguish_arr.__len__()):
        if(distinguish_arr[i] == 1):
            temp_arr.append(row[i])
    print(temp_arr)
    # print(temp_arr)
    result = net.activate(temp_arr)[0]
    if(result>0.5):
        print("TypeB")
        if(queries.irow(index2)[17] == 1):
            print("Right")
            right_num = right_num + 1
        else:
            print("Wrong")
            wrong_num = wrong_num + 1
    else:
        print("TypeA")
        if (queries.irow(index2)[17] == 0):
            print("Right")
            right_num = right_num + 1
        else:
            print("Wrong")
            wrong_num = wrong_num + 1
print("hit ratio: ", right_num/(right_num+wrong_num))