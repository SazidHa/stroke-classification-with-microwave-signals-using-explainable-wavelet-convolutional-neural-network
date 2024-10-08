# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:15:34 2024

@author: s4709145
"""


#%%
import pandas as pd;
import numpy as np;
import skrf as rf;
import os;
import matplotlib.pyplot as plt
import pywt
import pywt.data
import random
import h5py 


new_freq = rf.Frequency(0.5,2,1001,'ghz')
ind_all = (np.array([range(256)]))
ind_all = np.reshape(ind_all, (16, 16))
ind = ind_all[np.triu_indices(16,0)]
freq=sparameter.f

# if the files are not shorted sequentially and loaded as text order, this file method is to make the files sequential order
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


file_address=''
directory =os.chdir(file_address)
listoffile=sorted(os.listdir(directory)) 
listoffile.sort(key=natural_keys)

import glob
txtfiles = []
for file in glob.glob("C:\\Torso\\Data\\4tissueTorsoModel_horn_2ant_Debye_FDS\\1\\*/*.s2p"):
    txtfiles.append((file))
stotal1=[]
for data_file in ((listoffile)):
    sparameter = rf.Network(data_file)
    # sparameter=sparameter.interpolate(new_freq,coords='cart', kind = 'slinear')
    d=sparameter.s
    #reshapeddata=np.reshape(d, (49, 256))
    reshapeddata=np.reshape(d, (1001, 4))
    #reshapeddata=reshapeddata[:,ind]
    stotal1.append(reshapeddata)
stotal1=np.array(stotal1)

#for Haemorhegic data load from s2p file.
directory =os.chdir('C:\\Data store\\2-D data\\Single small target\\Hae')
#directory =os.chdir('C:\\Data store\\20210722-AS-AA-KSB-3B-SSBTarget\\HM\\CalibratedData')
listoffile=sorted(os.listdir(directory)) 
stotal2=[]
for data_file in ((listoffile)):
    sparameter = rf.Network(data_file)
    sparameter=sparameter.interpolate(new_freq,coords='cart', kind = 'slinear')
    d=sparameter.s
    reshapeddata=np.reshape(d, (401, 256))
    reshapeddata=reshapeddata[:,ind]
    stotal3.append(reshapeddata)
    
directory =os.chdir('C:\\Data store\\2-D data\\Single small target\\Isc')
#directory =os.chdir('C:\\Data store\\20210722-AS-AA-KSB-3B-SSBTarget\\IM\\CalibratedData')
listoffile=sorted(os.listdir(directory)) 

stotal3=[]
for data_file in ((listoffile)):
    sparameter = rf.Network(data_file)
    sparameter=sparameter.interpolate(new_freq,coords='cart', kind = 'slinear')
    d=sparameter.s
    reshapeddata=np.reshape(d, (401, 256))
    reshapeddata=reshapeddata[:,ind]
    stotal4.append(reshapeddata)

stotal=np.concatenate((stotal2, stotal3))

hf = h5py.File('C:\\Torso\\Data\\4tissueTorsoModel_horn_2ant_Debye_FDS\\Processed Data\\Frequency_domain_data.h5', 'w')
hf.create_dataset('Frequency_domain_data', data=stotal)
hf.close()