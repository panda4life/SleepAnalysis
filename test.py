from sampleParameters import sample
from clustering import clusteredData
import scipy as sp

data = sp.genfromtxt('../EEG_Data/xaa.txt')
amp = data[:,1]
s = sample(amp, 100.0, bands=sp.array([ [5,10] , [0,5] ]) )

#sampleParameters has been fully tested for correctness

