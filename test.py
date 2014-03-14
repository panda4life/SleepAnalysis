from sampleParameters import sample
from clustering import clusteredData
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# data = sp.genfromtxt('../EEG_Data/xaa.txt')
data = sp.genfromtxt('../EEG_Data/SC4001E0-PSG_data.txt')
sampleRate = 100  # Hz
# amp = data[:,1]
# s = sample(amp, 100.0, bands=sp.array([ [5,10] , [0,5] ]) )
# sampleParameters has been fully tested for correctness


clusterProc = clusteredData(3)  # 3 dimensional clustered eigenvector
blockSize = 3000
# take 2 second samples with 1 second overlap
for i in np.arange(0, data.shape[0] - 2 * sampleRate, sampleRate):
    dataSelect = np.arange(i, i + 2 * sampleRate)
    x = data[dataSelect, 1]
    clusterProc.appendSample(x, sampleRate, data[i,0])
    if((i/sampleRate)%blockSize == 0 and i != 0):
        clusterProc.analyze(blockSize,data[i,0])

fig1 = plt.figure()
ax = fig1.add_subplot(111,projection='3d')
ax.scatter(clusterProc.pc_eigen[:,0],clusterProc.pc_eigen[:,1],clusterProc.pc_eigen[:,2])
print(clusterProc.pc_eigen.shape)

fig2 = plt.figure()
plt.plot(data[:,0],data[:,1])
"""
clusterProc.analyze()
fig2 = plt.figure()
ax = fig2.add_subplot(111,projection='3d')
ax.scatter(clusterProc.pc_eigen[:,0],clusterProc.pc_eigen[:,1],clusterProc.pc_eigen[:,2])
print(clusterProc.pc_eigen.shape)
plt.show()
"""