import os
import h5py
output1 = 'train_color/1_F/train.h5'
output2 = 'train_pair/pair2/train.h5'
f1 = h5py.File(output1, 'r') 
f2 = h5py.File(output2, 'r') 
index = 222
print f1['landmark'][66]
print len(f2['eyedist'])
#for i in range(len(f1['landmark'])):
#   print f1['landmark'][i] - f2['landmark'][i]
#   print f1['eyedist'][i] - f2['eyedist'][i]
