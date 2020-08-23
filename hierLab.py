from math import sqrt
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import collections

# FUNCTION DECLARATIONS

def distCalc(dataSet):
    """
        Proximity matrix enumeration using basic distance function.
            dataSet - Pandas dataframe of input txt file.
    """
    xVals = dataSet['Att1'].values.tolist()
    yVals = dataSet['Att2'].values.tolist()
    zVals = dataSet['Att3'].values.tolist()
    qVals = dataSet['Att4'].values.tolist()
    distance = np.empty([len(xVals), len(yVals)], dtype = np.float128)

    for i in range(len(xVals)):
        for j in range(len(yVals)):
            value = sqrt((xVals[i]-xVals[j])**2 + (yVals[i]-yVals[j])**2 + (zVals[i]-zVals[j])**2 + (qVals[i]-qVals[j])**2)
            distance[i, j] = value
    return distance

def flatten(l):
    """
        Reads in irregular list, and outputs a generator object for flattening.
            l - irregular list of nd depth
    """
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def aver(x, y):
    return (x+y) / 2

def centr(x, y, orig, xlen, ylen):
    total = xlen + ylen
    xterm = (xlen / total) * x
    yterm = (ylen / total) * y
    zterm = ((xlen * ylen) / total**2) * orig
    return xterm + yterm - zterm

# FILE READS
path = './data'
all_files = glob.glob(path + "/*.txt")

dataSet = pd.read_csv('./data/Hierarchical_irisTesting.csv', sep=",", header=None, names=["Att1", "Att2", "Att3", "Att4"])

# Distance calculation method choice
funcs = [min, max, aver, centr]
names = ['min', 'max', 'aver', 'centr']
chosen = 2

# MAIN LOGIC
splits = [[i] for i in range(dataSet.shape[0])]
distance = distCalc(dataSet)

while len(splits) > 3:
    combine = np.where(distance == np.amin(distance[np.nonzero(distance)]))
    combine = [combine[0][0], combine[1][0]]
    if chosen == 3:
        xlen = 1 if len(splits[combine[0]]) == 1 else len(list(flatten(splits[combine[0]])))
        ylen = 1 if len(splits[combine[1]]) == 1 else len(list(flatten(splits[combine[1]])))
    splits[combine[0]].append([splits[combine[1]]])
    del splits[combine[1]]
    for i in range(distance.shape[0]):
        if distance[combine[0], i] == 0 or distance[combine[1], i] == 0:
            distance[combine[0], i] = 0
        else:
            if chosen == 3:
                orig = distance[combine[0], combine[1]]
                distance[combine[0], i] = funcs[chosen](distance[combine[0], i], distance[combine[1], i], orig, xlen, ylen)
                distance[i, combine[0]] = distance[combine[0], i]
            else:
                distance[combine[0], i] = funcs[chosen](distance[combine[0], i], distance[combine[1], i])
                distance[i, combine[0]] = distance[combine[0], i]
    distance = np.delete(distance, (combine[1]), axis=0)
    distance = np.delete(distance, (combine[1]), axis=1)

file = open('iris.txt', 'w')
output = np.zeros(50, dtype = int)

for i in range(len(splits)):
    split = list(flatten(splits[i]))
    for ind in split:
        output[ind] = (i+1)

for elem in output:
    file.write(str(elem))

# xVals = dataSet['Att1'].values.tolist()
# yVals = dataSet['Att2'].values.tolist()
# colors = ['red', 'blue']

# for i in range(len(splits)):
#     split = list(flatten(splits[i]))
#     splitX = [xVals[i] for i in split]
#     splitY = [yVals[i] for i in split]
#     plt.scatter(splitX, splitY, label='Split '+str(i), color=colors[i], s = 2)

# plt.legend(loc='upper left')
# plt.savefig('graphs/' + names[chosen] + '/iris.png')
# plt.title('Iris Dataset')
# plt.clf()
