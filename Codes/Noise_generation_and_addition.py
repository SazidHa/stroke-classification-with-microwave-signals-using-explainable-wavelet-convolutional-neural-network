# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:00:35 2024

@author: s4709145
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 21:42:54 2022

@author: s4709145
"""
import pandas as pd;
import numpy as np;
import skrf as rf;
import os;
import matplotlib.pyplot as plt
import pywt
import pywt.data
import h5py 
import matplotlib.pyplot as plt
from keras.models import Sequential
import ssl
import pandas as pd;
import numpy as np;
import skrf as rf;
import os;
import matplotlib.pyplot as plt
import pywt
import pywt.data
import h5py 
#%%
hf = h5py.File("C:\\Data store\\2-D data\\All_time_domain_data\\dataforalltest.h5", 'r')
# hf.keys()
n1 = hf.get('dataforalltest')
n1 = np.array(n1)
n1.shape
hf.close()

hf = h5py.File("C:\\Data store\\2-D data\\All_time_domain_data\\ydataforalltest.h5", 'r')
# hf.keys()
n2 = hf.get('ydataforalltest')
n2 = np.array(n2)
n2.shape
hf.close()

hf = h5py.File("C:\\Data store\\2-D data\\Calibrated data\\subtractedstdvalue.h5", 'r')
# hf.keys()
n3 = hf.get('dataset1')
n3 = np.array(n3)
n3.shape
hf.close()
calibratedstd=n3.T

hf = h5py.File("C:\\Data store\\2-D data\\Calibrated data\\subtractedmeanvaluereal.h5", 'r')
# hf.keys()
n4 = hf.get('dataset1')
n4 = np.array(n4)
n4.shape
hf.close()
calibratedmeanreal=n4.T


hf = h5py.File("C:\\Data store\\2-D data\\Calibrated data\\subtractedmeanvalueimag.h5", 'r')
# hf.keys()
n5 = hf.get('dataset1')
n5 = np.array(n5)
n5.shape
hf.close()
calibratedmeanimag=n5.T

hf = h5py.File("C:\\Data store\\2-D data\\VNA and Head movement data\\stddevrealpatient.h5", 'r')
# hf.keys()
n6 = hf.get('dataset1')
n6 = np.array(n6)
n6.shape
hf.close()
realpatientstd=n6.T

hf = h5py.File("C:\\Data store\\2-D data\\VNA and Head movement data\\meanvaluepatientreal.h5", 'r')
# hf.keys()
n7 = hf.get('dataset1')
n7 = np.array(n7)
n7.shape
hf.close()
realpatientmeanreal=n7.T

hf = h5py.File("C:\\Data store\\2-D data\\VNA and Head movement data\\meanvaluepatientimag.h5", 'r')
# hf.keys()
n8 = hf.get('dataset1')
n8 = np.array(n8)
n8.shape
hf.close()
realpatientmeanimag=n8.T

data_with_noise_real=np.zeros(shape=(9000,49,256))
data_with_noise_imag=np.zeros(shape=(9000,49,256))
data_with_noise_real1=np.zeros(shape=(9000,49,256))
data_with_noise_imag1=np.zeros(shape=(9000,49,256))
data_with_noise=np.zeros(shape=(9000,49,256))
noise_real=np.zeros(shape=(9000,49,256))
noise_imag=np.zeros(shape=(9000,49,256))
Stddev=np.zeros(shape=(1,49,256))
mureal=np.zeros(shape=(1,49,256))
muimg=np.zeros(shape=(1,49,256))
#%%
# generation and addition VNA and head movement noise 
import scipy.stats as stats
from scipy.stats import truncnorm
for signal in range(len(realpatientmeanreal[0])):
    for freq in range(len(realpatientmeanreal)):
        # a,b=minvalue[freq,signal],maxnvalue[freq,signal]
        mu1,mu2,sigma=realpatientmeanreal[freq,signal],realpatientmeanimag[freq,signal],realpatientstd[freq,signal]
        dist1=stats.truncnorm(-4,4,loc=mu1,scale=sigma)
        dist2=stats.truncnorm(-4,4,loc=mu2,scale=sigma)
        noise1=dist1.rvs(9000)
        noise2=dist2.rvs(9000)
        Stddev[:,freq,signal]=Stddev[:,freq,signal]+sigma
        mureal[:,freq,signal]=mureal[:,freq,signal]+mu1
        muimg[:,freq,signal]=muimg[:,freq,signal]+mu2
        noise_real[:,freq,signal]=noise_real[:,freq,signal]+noise1
        noise_imag[:,freq,signal]=noise_imag[:,freq,signal]+noise2
        data_with_noise_real[:,freq,signal]=n1[:,freq,signal].real+noise1
        data_with_noise_imag[:,freq,signal]=n1[:,freq,signal].imag+noise2

#%%
#plotting the VNA and head movement noise 
ind_all = (np.array([range(256)]))
ind_all = np.reshape(ind_all, (16, 16))
ind = ind_all[np.triu_indices(16,3)]
noise_real=noise_real[:,:,ind]
noise_imag=noise_imag[:,:,ind]
Stddev=Stddev[:,:,ind]
mureal=mureal[:,:,ind]
muimg=muimg[:,:,ind]
total_mu=(abs(mureal+1j*muimg))
total_noise=20*np.log10(abs(noise_real+1j*noise_imag))
dbstddev=abs(Stddev)
negtiveestd=total_mu+dbstddev
a=np.linspace(0.5,2,49)
positivestddev=20*np.log10(dbstddev)
dbmu=20*np.log10(total_mu)
negativeestd=(dbmu+positivestddev)/2
plt.figure(dpi=300)
#for i in range(len(total_noise[1][1])):
    
plt.plot(a, dbmu[0,:,2],'b--',label='+4 Std deviation' )

plt.plot(a,positivestddev[0,:,2],'r--',label=' -4 Std deviation')
 
plt.plot(a,negativeestd[0,:,2],'y', label='mean')
plt.legend()
    #plt.ylim(-85,-19)
plt.title('Noise distribution for a signal over frequency range',fontsize=13)
plt.ylabel('Power(dB)',fontsize=12)
plt.xlabel('frequency(GHz)',fontsize=12)

#plt.tight_layout()

plt.show()

#plt.plot(dbstddev[0,:,0])
#%%
# Adding antenna Dissimilarity Noise
noise3_real=np.zeros(shape=(9000,49,128))
noise4_imag=np.zeros(shape=(9000,49,128))
Stddev2=np.zeros(shape=(1,49,128))
for signal in range(len(calibratedstd[0])):
    for freq in range(len(calibratedstd)):
        # a,b=minvalue[freq,signal],maxnvalue[freq,signal]
        mu3,mu4,sigma=calibratedmeanreal[freq,signal],calibratedmeanimag[freq,signal],calibratedstd[freq,signal]
        noise3=[mu3]*9000
        noise4=[mu4]*9000
        Stddev2[:,freq,signal]=Stddev2[:,freq,signal]+sigma
        noise3_real[:,freq,signal]=noise3_real[:,freq,signal]+ noise3
        noise4_imag[:,freq,signal]=noise4_imag[:,freq,signal]+ noise4
        data_with_noise_real[:,freq,signal]=data_with_noise_real[:,freq,signal].real+noise3
        data_with_noise_imag[:,freq,signal]=data_with_noise_imag[:,freq,signal].imag+noise4
#%%
Calibrated_noisedata=data_with_noise_real+1j*data_with_noise_imag
Calibrated_noisedata=Calibrated_noisedata[:,:,ind]
totalcalibratednoise=np.where(abs(Calibrated_noisedata) == np.amax(abs(Calibrated_noisedata.max())))
totalcalibratednoisedb=np.where(totalcalibratednoise == np.amax(totalcalibratednoise))
#%%
total_mu1=(abs(mureal+1j*muimg))
total_noise1=20*np.log10(abs(noise3_real+1j*noise4_imag))
totaldatawitnoise=data_with_noise_real+1j*data_with_noise_imag
totalcalibratednoise=totaldatawitnoise[:,:,ind]
totalcalibratednoisedb=20*np.log10(totalcalibratednoise.max())
dbstddev1=abs(Stddev2)
negtiveestd1=total_mu1+dbstddev1
a1=np.linspace(0.5,2,49)
positivestddev1=20*np.log10(dbstddev1)
dbmu1=20*np.log10(total_mu1)
negativeestd1=(dbmu1+positivestddev1)/2
plt.figure(dpi=300)
for i in range(len(total_noise1[1][1])):
    
    plt.plot(a, dbmu1[0,:,1],'b--')

    plt.plot(a,positivestddev1[0,:,1],'r--')
 
    plt.plot(a,negativeestd1[0,:,1],'y')
    
    #plt.ylim(-85,-19)
plt.title('Noise distribution for a signal over frequency range',fontsize=13)
plt.ylabel('Power(dB)',fontsize=12)
plt.xlabel('frequency(GHz)',fontsize=12)
plt.tight_layout()
plt.show()
 
#%%
data_with_noise=data_with_noise_real+1j*data_with_noise_imag 
realnoise=data_with_noise_real[:,:,ind]-n1.real[:,:,ind]
imagnoise=data_with_noise_imag[:,:,ind]-n1.imag[:,:,ind]
totalnoiseindb=20*np.log10(abs(realnoise+1j*imagnoise))
ind_all = (np.array([range(256)]))
ind_all = np.reshape(ind_all, (16, 16))
ind = ind_all[np.triu_indices(16,3)]
Total_rawdata=n1[:,:,ind]
total_noisedata=data_with_noise[:,:,ind]
Combinenoise_for_alldata=np.fft.irfft(Total_data,axis=1)
 
hf = h5py.File('C:\\Data store\\Combine noise\\withoutnoisedata.h5', 'w')
hf.create_dataset('withoutnoisedata', data=Combinenoise_for_alldata)
hf.close()
hf = h5py.File('C:\\Data store\\Combine nois\\calibratedydata.h5', 'w')
hf.create_dataset('calibratedydata', data=y_total)
hf.close()
hf = h5py.File('C:\\Data store\\Calibrated data\\calibratedYindc.h5', 'w')
hf.create_dataset('calibratedYindc', data=y_indctotal)
hf.close()

