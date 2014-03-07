from sampleParameters import sample
import scipy as sp

data = sp.genfromtxt('../EEG_Data/xaa.txt')
amp = data[:,1]
s = sample(amp, 100.0, bands=sp.array([ [5,10] , [0,5] ]) )

# good
# print "Activity:", s.activity(s.x)

# good
# print "Mobility:", s.mobility(s.x)

# good
# print "Complexity:", s.complexity(s.x)

# untested: function "find"
# print s.harmonic(s.x)

print s.fft_bandpower(s.x)