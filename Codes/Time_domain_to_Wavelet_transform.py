# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:52:44 2023

@author: s4709145
"""

"""
Created on Thu Feb 17 09:05:15 2022

@author: s4709145
"""
#import libraries
import pandas as pd;
import numpy as np;
import skrf as rf;
import os;
import pywt
import pywt.data
from skimage.transform import resize
import h5py
from pprint import pprint as pp
#%%
#Load time domain data


hf = h5py.File("C:\\Torso\\Data\\New Data\\Time data and CWT data\\Torso_time_data_2.h5", 'r')
# hf.keys()
X = hf.get('Torso_time_data_2')
X= np.array(X)
X.shape
hf.close()
X=X.T
X=np.reshape(X, (241,300,1)) #transpose the data to make the shape (case number*time point*channel)


#%%
def create_cwt_images(X, n_scales, rescale_size, wavelet_name = "morl"):
    n_samples = X.shape[0] 
    n_signals = X.shape[2] 
    
    # range of scales that fits the frequency range 0.5 GHz to 2 GHz
    scales =[21.6666666666667,18.0555555555556,15.4761904761905,13.5416666666667,12.0370370370370,10.8333333333333,9.84848484848485,9.02777777777778,8.33333333333334,7.73809523809524,7.22222222222222,6.77083333333333,6.37254901960784,6.01851851851852,5.70175438596491,5.41666666666667]
    # pre allocate array
    X_cwt = np.ndarray(shape=(n_samples,n_scales,rescale_size, n_signals,),dtype='float32')
    
    for sample in range(n_samples):
        if sample % 1000 == 0:
            print(sample)
        for signal in range(n_signals):
            serie = X[sample, :, signal]
            # continuous wavelet transform 
            coeffs, freqs = pywt.cwt(serie, scales, wavelet_name,sampling_period=0.75e-10)
            #resize the 2D cwt coeffs
            rescale_coeffs = resize(coeffs, (n_scales,rescale_size ),mode='constant')
            X_cwt[sample,:,:,signal] = rescale_coeffs 
            
            
    return X_cwt,freqs
  
# amount of pixels in X and Y 
rescale_size = 68
# determine the max scale size
n_scales = 16

X_cwt,freqs = create_cwt_images(X, n_scales, rescale_size)
#print(f"shapes (n_samples, x_img, y_img, z_img) of data_wavelet: {X__cwt.shape}")

#%%
#save the wavelet feature in h5py format.
hf = h5py.File('C:\\Torso\\Data\\New Data\\Time data and CWT data\\Torso_06-1.3Ghz_cwt_2_edited.h5','w')
hf.create_dataset('Torso_06-1.3Ghz_cwt_2_edited', data=X_cwt)
hf.close()