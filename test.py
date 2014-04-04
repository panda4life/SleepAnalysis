
from sampleParameters import sample
from clustering import clusteredData
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pyedf as edf
from mpl_toolkits.mplot3d import Axes3D

import sys

# data = sp.genfromtxt('../EEG_Data/xaa.txt')
#data = sp.genfromtxt('../EEG_Data/SC4001E0-PSG_data.txt')
#sampleRate = 100  # Hz

annotations = []

# 50-50 split of first half of SC4001E0-PSG_data.txt respectively
training_data = "../EEG_data/training.txt"
testing_data = "../EEG_data/testing.txt"

print('Loading annotation file')
with open("../EEG_data/SC4001EC-Hypnogram_annotations.txt") as f:
    header = f.readline()
    for line in f:
        line = line.strip()
        columns = line.split(',')
        stageNo = -1

        # consolidating (R,1)=1, (2)=2, (3,4)=3, (W)=4
        if(columns[2] == 'Sleep stage 1'):
            stageNo = 1
        elif(columns[2] == 'Sleep stage 2'):
            stageNo = 2
        elif(columns[2] == 'Sleep stage 3'):
            stageNo = 3
        elif(columns[2] == 'Sleep stage 4'):
            stageNo = 3
        elif(columns[2] == 'Sleep stage R'):
            stageNo = 1
        elif(columns[2] == 'Sleep stage W'):
            stageNo = 0
        else:
            pass #keep the stageNo at -1
        if(annotations == []):
            annotations = [float(columns[0]),float(columns[1]),stageNo]
        else:
            dataVec = [float(columns[0]),float(columns[1]),stageNo]
            annotations = np.vstack((annotations,dataVec))
        #print(data[-1,:])
print('Opening File Complete') 



def grabAnnotation(time):
    annotationIndex = np.where(annotations[:,0] < time)
    annotationIndex = annotationIndex[0][-1]
    stage = annotations[annotationIndex][2]
    return stage


def loadData(filename, clusterProc, sampleRate=100, sampleDuration=30, sampleOverlap=15, numLines=400000, doAnnotations=False):
    print('Loading data file')
    data = []
    with open(filename) as f:
        header = f.readline()
        linecount = 0
        # avoid reading whole file into memory
        for line in f:
            line = line.strip()
            columns = line.split(',')
            # first run
            if(data == []):
                data = [float(columns[0]),float(columns[1]),float(columns[2])]
            else:
                dataVec = [float(columns[0]),float(columns[1]),float(columns[2])]
                data = np.vstack((data,dataVec))
            linecount += 1
            # when buffer reaches sampleDuration keep overlap for the next sampling and drop the data we're done with
            if(len(data) == sampleDuration*sampleRate):
                data_tstamp = data[0,0]
                anno = -1
                if doAnnotations:
                    anno = grabAnnotation(data_tstamp)
                clusterProc.appendSample(data[:,1], sampleRate, data_tstamp, anno)
                data = data[(-sampleOverlap)*sampleRate:]
            if(linecount%10000 == 0):
                print('' + str(linecount) + '\tlines loaded')
            if(linecount > numLines):
                break
    print('Opening File Complete')

print('Declare Initial Cluster Processor')


# with open("../EEG_data/SC4001E0-PSG_data.txt") as f:
clusterProc = clusteredData(3)  # 3 dimensional clustered eigenvector
loadData(training_data, clusterProc, doAnnotations=True)
clusterProc.initialLearn()
# print clusterProc.kmeansDist(1, 2)
# sys.exit(1)

loadData(testing_data, clusterProc, doAnnotations=False)
clusterProc.classifyNewSamples()

dummyCluster = clusteredData(3)
loadData(testing_data,dummyCluster, doAnnotations=True)
dummyCluster.initialLearn()


countSimilarities = 0

dummyLen = len(dummyCluster.samples)
for i in range(0, dummyLen):
    assert(clusterProc.sp_tstamp[i+dummyLen] == dummyCluster.sp_tstamp[i])
    
    print "Asserted/Correct:", clusterProc.samples[i+dummyLen].category, dummyCluster.samples[i].category

    if(clusterProc.samples[i+dummyLen].category == dummyCluster.samples[i].category):
        countSimilarities += 1
print(float(countSimilarities)/dummyLen*100)
# amp = data[:,1]
# s = sample(amp, 100.0, bands=sp.array([ [5,10] , [0,5] ]) )
# sampleParameters has been fully tested for correctness



# print('Start cluster analysis')
"""
for i in np.arange(0, data.shape[0] - 30 * sampleRate, 15*sampleRate):
    dataSelect = np.arange(i, i + 30 * sampleRate)
    x = data[dataSelect, 1]
"""    

"""
clusterProc.analyze()

fig1 = plt.figure()
ax = fig1.add_subplot(111,projection='3d')
for i in np.arange(0,clusterProc.pc_eigen.shape[0]):
    annotationIndex = np.where(annotations[:,0] < clusterProc.sp_tstamp[i])[0][-1]
    stage = annotations[annotationIndex,2]
    if(stage == 0): #Waking
        ax.scatter(clusterProc.pc_eigen[i,0],clusterProc.pc_eigen[i,1],clusterProc.pc_eigen[i,2], color="black", label='Wake', s = 2)
    elif(stage == 1): #Stage 1
        ax.scatter(clusterProc.pc_eigen[i,0],clusterProc.pc_eigen[i,1],clusterProc.pc_eigen[i,2], color="red", label='light', s = 2)
    elif(stage == 2): #Stage 2
        ax.scatter(clusterProc.pc_eigen[i,0],clusterProc.pc_eigen[i,1],clusterProc.pc_eigen[i,2], color="purple", label='light', s = 2)
    elif(stage == 3): #Stage 3
        ax.scatter(clusterProc.pc_eigen[i,0],clusterProc.pc_eigen[i,1],clusterProc.pc_eigen[i,2], color="blue", label='deep', s = 2)
    elif(stage == 4): #Stage 4
        ax.scatter(clusterProc.pc_eigen[i,0],clusterProc.pc_eigen[i,1],clusterProc.pc_eigen[i,2], color="blue", label='deep', s = 2)
    elif(stage == 5): #Stage 5
        ax.scatter(clusterProc.pc_eigen[i,0],clusterProc.pc_eigen[i,1],clusterProc.pc_eigen[i,2], color="green", label='REM', s = 2)
    else:
        pass
ax.set_title('Clustered Sleep Eigenvalues after KPCA dimensionality reduction')
ax.set_xlabel('1st Eigenvalue')
ax.set_ylabel('2nd Eigenvalue')
ax.set_zlabel('3rd Eigenvalue')
# plt.legend(loc='upper left')
print(clusterProc.pc_eigen.shape)

#fig2 = plt.figure()
#plt.plot(data[:,0],data[:,1])


clusterProc.analyze()
fig2 = plt.figure()
ax = fig2.add_subplot(111,projection='3d')
ax.scatter(clusterProc.pc_eigen[:,0],clusterProc.pc_eigen[:,1],clusterProc.pc_eigen[:,2])
print(clusterProc.pc_eigen.shape)
plt.show()

"""
