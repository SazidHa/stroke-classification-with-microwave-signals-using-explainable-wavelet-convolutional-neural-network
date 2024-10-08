# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:57:52 2024

@author: s4709145
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 12:29:39 2022

@author: s4709145
"""
import shap
import numpy as np
import tensorflow as tf
from tensorflow import keras
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
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import random
import tensorflow.keras.backend as K
#%%
hf = h5py.File("C:\\CNN Clasification\\3-D data\\3d Withoutnoise train test results\\16scalewith32scale\\Xtrainwithout3d16scale32point256.h5", 'r')
# hf.keys()
n3 = hf.get('Xtrainwithout3d16scale32point256')
n3 = np.array(n3)
n3.shape
hf.close()

hf = h5py.File("C:\\CNN Clasification\\3-D data\\3d Withoutnoise train test results\\16scalewith32scale\\Xtestwithout3d16scale32point256.h5", 'r')
# hf.keys()
n2 = hf.get('Xtestwithout3d16scale32point256')
n2 = np.array(n2)
n2.shape
hf.close()

hf = h5py.File("C:\\CNN Clasification\\3-D data\\3d Withoutnoise train test results\\16scalewith32scale\\ytestwithout3d16scale32point256.h5", 'r')
# hf.keys()
n4 = hf.get('ytestwithout3d16scale32point256')
n4 = np.array(n4)
n4.shape
hf.close()
#testdata=n2[[360,440,454,850,931],:,:,:]

from numpy import loadtxt
from keras.models import load_model
model = load_model("C:\\CNN Clasification\\3-D data\\3d Withoutnoise train test results\\16scalewith32scale\\best_model_without16scale32point256.h5")
#%%

y_pred_proba = model.predict(testdata)
# select a set of background examples to take an expectation over
background = n3[np.random.choice(n3.shape[0], 150, replace=False)]
# DeepExplainer to explain predictions of the model
e=shap.DeepExplainer(model, background)
a=np.arange(1,17)
scale=np.array(pywt.scale2frequency('morl', a)/0.98e-10)

scale=np.reshape(scale,(16,1))
list1 = scale.tolist()
# compute shap values
shap_values = e.shap_values(n2)
shap_values =np.array(shap_values)[0]
#%%
averagevalue=np.zeros(shape=(28,2548))
#yzero=(n5[0:500]==0)
#zeroposition = [i for i, val in enumerate(yzero) if val]
#for idx in range(len(zeroposition)):
#    averagevalue=averagevalue+l[0,zeroposition[idx],:,:,0]
#haemorhagicmean=averagevalue/len(zeroposition);
maximumshap=shap_values[0].max()
result= (np.array(np.where(shap_values[0] == np.amax(shap_values[0]))).T)
#%%
maximumshap=np.zeros(shape=(169,1))
resultoutcome=[]
for i in range(len(shap_values)):
    maximumshap[i]=shap_values[i].max()
    result= np.asarray(np.where(shap_values[i] == np.amax(shap_values[i]))).T
    resultoutcome.append(result)
output=np.asarray(resultoutcome)
#%%
averagevalue=np.zeros(shape=(16,32))
for idx in range(256):
       averagevalue=averagevalue+shap_values[1,:,:,i]
averagevalue=averagevalue/256
fig, ax = plt.subplots(1,1)
plt.imshow(averagevalue,extent=[0,30,15,0],aspect='auto')
y_label_list = ['8.29e+09','2.76e+09','1.65e+09', '1.18e+09', '9.21e+08', '7.53e+08', '6.37e+08', '5.52e+08']
ax.set_yticklabels(y_label_list)
plt.colorbar()
#%%
fig, ax = plt.subplots(1,1)
plt.imshow(shap_values[164,:,:,68],extent=[0,30,15,0],aspect='auto')
y_label_list = ['8.29e+09','2.76e+09','1.65e+09', '1.18e+09', '9.21e+08', '7.53e+08', '6.37e+08', '5.52e+08']
ax.set_yticklabels(y_label_list)
plt.colorbar()

#y_positions =np.reshape(a,(16,1)) # pixel count at label position

#plt.imshow(n2[0,:,:,165])
#avergevalue=np.zeros(shape=(28,2548))
#yone=(n5[0:500]==1)
#%%
plt.imshow(shap_values[1,:,:,15])
def image(shap_values, pixel_values=None, labels=None, width=20, aspect=0.2, hspace=0.2, labelpad=None, show=True, plotchannels=[0]):
     """ Plots SHAP values for image inputs.
     Parameters
     ----------
     shap_values : [numpy.array]
         List of arrays of SHAP values. Each array has the shap (# samples x width x height x channels), and the
         length of the list is equal to the number of model outputs that are being explained.
     pixel_values : numpy.array
         Matrix of pixel values (# samples x width x height x channels) for each image. It should be the same
         shape as each array in the shap_values list of arrays.
     labels : list
         List of names for each of the model outputs that are being explained. This list should be the same length
         as the shap_values list.
     width : float
         The width of the produced matplotlib plot.
     labelpad : float
         How much padding to use around the model output labels.
     show : bool
         Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot
         to be customized further after it has been created.
     """

     # support passing an explanation object
     if str(type(shap_values)).endswith("Explanation'>"):
         shap_exp = shap_values
         feature_names = [shap_exp.feature_names]
         ind = 0
         if len(shap_exp.base_values.shape) == 2:
             shap_values = [shap_exp.values[..., i] for i in range(shap_exp.values.shape[-1])]
         else:
             raise Exception("Number of outputs needs to have support added!! (probably a simple fix)")
         if pixel_values is None:
             pixel_values = shap_exp.data
         if labels is None:
             labels = shap_exp.output_names

     multi_output = True
     if type(shap_values) != list:
         multi_output = False
         shap_values = [shap_values]

     # make sure the number of channels to plot is <= number of channels present
     nchannels = min(shap_values[0].shape[-1], len(plotchannels))
     plotchannels = plotchannels[:nchannels]

     # make sure labels
     if labels is not None:
         labels = np.array(labels)
         assert labels.shape[0] == shap_values[0].shape[0], "Labels must have same row count as shap_values arrays!"
         if multi_output:
             assert labels.shape[1] == len(shap_values), "Labels must have a column for each output in shap_values!"
         else:
             assert len(labels.shape) == 1, "Labels must be a vector for single output shap_values."

     label_kwargs = {} if labelpad is None else {'pad': labelpad}

     # plot our explanations
     x = pixel_values
     fig_size = np.array([3 * (len(shap_values) + 1), 2.5 * (x.shape[0] + nchannels)])
     if fig_size[0] > width:
         fig_size *= width / fig_size[0]
     fig, axes = pl.subplots(nrows=x.shape[0] * nchannels, ncols=len(shap_values) + 1, figsize=fig_size)
     if len(axes.shape) == 1:
         axes = axes.reshape(1,axes.size)
     for row in range(x.shape[0]):
         x_curr = x[row].copy()

         # make sure we have a 2D array for grayscale
         if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
             x_curr = x_curr.reshape(x_curr.shape[:2])
         if x_curr.max() > 1:
             x_curr /= 255.

         # get a grayscale version of the image
         if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
             x_curr_gray = (0.2989 * x_curr[:,:,0] + 0.5870 * x_curr[:,:,1] + 0.1140 * x_curr[:,:,2]) # rgb to gray
             x_curr_disp = x_curr
         elif len(x_curr.shape) == 3:
             x_curr_gray = x_curr.mean(2)

             # for non-RGB multi-channel data we show an RGB image where each of the three channels is a scaled k-mean center
             flat_vals = x_curr.reshape([x_curr.shape[0]*x_curr.shape[1], x_curr.shape[2]]).T
             flat_vals = (flat_vals.T - flat_vals.mean(1)).T
             means = kmeans(flat_vals, 3, round_values=False).data.T.reshape([x_curr.shape[0], x_curr.shape[1], 3])
             x_curr_disp = (means - np.percentile(means, 0.5, (0,1))) / (np.percentile(means, 99.5, (0,1)) - np.percentile(means, 1, (0,1)))
             x_curr_disp[x_curr_disp > 1] = 1
             x_curr_disp[x_curr_disp < 0] = 0
         else:
             x_curr_gray = x_curr
             x_curr_disp = x_curr

         axes[row*nchannels,0].imshow(x_curr_disp, cmap=pl.get_cmap('gray'))
         axes[row*nchannels,0].axis('off')
         for c in range(1, nchannels):
             axes[row*nchannels+c, 0].set_visible(False)

         if len(shap_values[0][row].shape) == 2:
             abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
         else:
             abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()
         max_val = np.nanpercentile(abs_vals, 99.9)
         for i in range(len(shap_values)):
             if labels is not None:
                 axes[row*nchannels,i+1].set_title(labels[row,i], **label_kwargs)

             for c in range(nchannels):
                 sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row][..., plotchannels[c]]
                 axes[row*nchannels+c,i+1].imshow(x_curr_gray, cmap=pl.get_cmap('gray'), alpha=0.15, extent=(-1, sv.shape[1], sv.shape[0], -1))
                 im = axes[row*nchannels+c,i+1].imshow(sv, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
                 axes[row*nchannels+c,i+1].axis('off')

     if hspace == 'auto':
         fig.tight_layout()
     else:
         fig.subplots_adjust(hspace=hspace)
     cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal", aspect=fig_size[0]/aspect)
     cb.outline.set_visible(False)
     if show:
         pl.show()

image(shap_values)
