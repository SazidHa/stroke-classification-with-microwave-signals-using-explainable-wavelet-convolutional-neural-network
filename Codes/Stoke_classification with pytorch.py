# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:07:38 2023

@author: s4709145
"""

import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit
import torch.nn.functional as F
#import pandas as pd;
import numpy as np;
import skrf as rf;
import os;
import matplotlib.pyplot as plt
#import pywt
#import pywt.data
#from skimage.transform import resize
import h5py

# device=torch.device('cpu')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hf = h5py.File("C:\\CNN Clasification\\3-D data\\Without Noise\\Total time domain and CWT data\\without16scale68point941with256cwt1.h5", 'r')
# hf.keys()
n3 = hf.get('without16scale68point941with256cwt1')
n3 = np.array(n3)
n3.shape
hf.close()
n1=n3
hf = h5py.File("C:\\CNN Clasification\\3-D data\\Without Noise\\Total time domain and CWT data\\940Ydataforall.h5", 'r')
# hf.keys()
n2 = hf.get('940Ydataforall')
n2 = np.array(n2)
n2.shape
hf.close()
i = [0,3,1,2]
 
# create output
n3= np.moveaxis(n1,3,1)

#X, y = shuffle(n3, n2)
X=torch.from_numpy(n3)
# n2 = np.c_[n2,1-n2] # wcl
y=n2.astype(np.float32)

X_train, X_test,y_train, y_test = train_test_split(X,y ,
                                   random_state=104, 
                                   test_size=0.20, 
                                   shuffle=True)
train_data = []
for i in range(len(X_train)):
   train_data.append([X_train[i], y_train[i]])
   
test_data = []
for i in range(len(X_test)):
   test_data.append([X_test[i], y_test[i]])

train_loader=DataLoader(train_data,batch_size=64,shuffle=False)
test_loader=DataLoader(test_data,batch_size=64,shuffle=False)

Classes={0:'Haemorhagic', 1: 'Ischemic'}
#Classes=list(Classes)
#device=torch.device('cpu')
#%%
# def init_weights(m):
#     if type(m)==nn.Linear or type(m)==nn.Conv2d:
#         torch.nn.init.xavier_uniform(m.weight)
        
class ConvNet(nn.Module):
    def __init__(self,num_classes=2):
        super(ConvNet,self).__init__()
        #output size after convolution filter
        #((w-f+2P)/s)+1
        
        self.conv1=nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding='same')
        #shape(256,12,150,150)
        self.bn1=nn.BatchNorm2d(num_features=128)
        self.relu1=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2)
        #shape(256,12,75,75)
        self.conv2=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding='same')
        
        #shape(256,20,75,75)
        self.relu2=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.dropout=nn.Dropout2d(p=0.1)
        #shape(256,20,75,75)
        #self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #shape(256,32,75,75)
        self.bn2=nn.BatchNorm2d(num_features=256)
        #self.relu3=nn.ReLU()
        #shape(256,32,75,75)
        self.relu3=nn.ReLU()
        self.relu4=nn.ReLU()
        
        self.fc1=nn.Linear(in_features=256*4*17, out_features=512)
        self.fc2=nn.Linear(in_features=512, out_features=128)
        self.fc3=nn.Linear(in_features=128, out_features=1) #[1,2] wcl check one-hot
        self.sigmoid=nn.Sigmoid()
        
        #initialization
        # nn.init.kaiming_normal(self.conv1.weight)

        nn.init.kaiming_normal(self.conv2.weight)
        nn.init.kaiming_normal(self.fc1.weight)
        nn.init.kaiming_normal(self.fc2.weight)
        nn.init.xavier_normal(self.fc3.weight)
        #self.apply(init_weights)
        
    def forward(self,input):
        l
        output=self.bn1(output)

        output=self.relu1(output)
        output=self.pool(output)
        output=self.conv2(output)
        output=self.bn2(output)
        output=self.relu2(output)

        output=self.pool(output)

            #output shape(256,32,75,75)
        output=output.view(-1,256*4*17)
        output=self.fc1(output)
        output=self.relu3(output)
        output=self.dropout(output)
        output=self.fc2(output)
        output=self.relu4(output)
        output=self.fc3(output)
        output=self.sigmoid(output)
        return output
        


#%%
model = ConvNet(num_classes=2).to(device)
model = model.cuda()
# optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
optimizer=Adam(model.parameters(),lr=0.00001)
loss_function=nn.BCELoss()

#loss_function=torch.nn.functional.binary_cross_entropy
num_epoch=200
train_count=len(train_loader)
test_count=len(test_loader)
#%%
torch.set_printoptions(precision=4)
for epoch in range(num_epoch):
    #Evaluate and tain on training dataset
    model.train()
    
    train_accuracy=0.0
    train_loss=0.0
    
    for i, (images,labels) in enumerate(train_loader):
        # images=images.to(device)
        # labels=labels.unsqueeze(1).to(device)
        images=images.cuda()
        labels=labels.unsqueeze(1).cuda()
        # labels=labels.cuda()

         
        # images=images.to(device)
        # labels=labels.unsqueeze(1).to(device)
        
        outputs=model(images)
        optimizer.zero_grad()
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        # train_loss+= loss.cpu().data*images.size(0)
        train_loss+= (loss).data.cpu()
        roundvalue=torch.round(outputs)
        _,prediction=torch.max(outputs,1)
        
        train_accuracy+= int(torch.sum(roundvalue==labels))/len(labels)
        
    # print(f'the device we are working on: {device}')
    #print(f'now finishing epoch:{epoch}')
    
    
    train_accuracy=train_accuracy/train_count
    print(f'now finishing epoch:{epoch}')
    #print('The train_loss : torch.round(loss,decimals=3)', 'and  train_accuracy : "{0:.4f}".format{train_accuracy}')
    print('The train_loss :' "{0:.5f}".format(loss), 'train_accuracy :' "{0:.5f}".format(train_accuracy))
    #print(f'The train_loss : round({loss}, 2)', f'and  train_accuracy : round({train_accuracy},2)')
    #print(f'and the train_accuracy now is: {train_accuracy}')
    with torch.no_grad():
        model.eval()
        test_accuracy=0.0
    
        for i, (images1,labels1) in enumerate(test_loader):
            
            images1=images1.cuda()
            labels1=labels1.unsqueeze(1).cuda()
        
            outputs1=model(images1)
        #_,prediction=torch.max(outputs.data,1)
            roundvalue1=torch.round(outputs1)
            test_accuracy+= int(torch.sum(roundvalue1==labels1.data))/len(labels1)
    
    test_accuracy=test_accuracy/test_count
    print('test_accuracy:' "{0:.4f}".format(test_accuracy))
    #print(f'and the test_accuracy now is: {test_accuracy}')
