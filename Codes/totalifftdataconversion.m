
clc;
close all;
clear all;

 
dirname = 'C:\Torso\Data\Realistic model Data\4tissueTorsoRealisticModel_horn_2ant_Debye_TDS\4tissueTorsoRealisticModel_horn_2ant_Debye_TDS-0001\All_results\';
file = dir([dirname,'*.s2p']);
files = {file.name};

folder= {file.folder};
freq=linspace(0.5e9,2e9,1001);
data=[];
timedomaindata=[];
haemorhagicdata=[];
ischemicdata=[];
%load all data from the library
for k = 1:length(file)
  filea = sparameters(strcat(folder{k},'\',files{k}));
  rows=length(filea.Frequencies);
  data1=reshape(permute(filea.Parameters,[3 2 1]),rows,[]); % reshape the data from 16*16*751 to 751*256
  data(k,:,:)= data1;
 end
for l=1:length(file)
  for k=1:256
    [timedomaindata(l,:,k), t]=freqtotime2(data(l,:,k),filea.Frequencies); %time domain conversion from frequency domain data.
    end
end
Strokedata(:,:,:)=timedomaindata(:,1:68,:); %68 point data are selected which shows major contribution of the signal

 
h5create('C:\CNN Clasification\3-D data\labdata751timedomain\mat751Buketlabpos3.h5','/mat751Buketlabpos3',size(Strokedata))
h5write('C:\CNN Clasification\3-D data\labdata751timedomain\mat751Buketlabpos3.h5','/mat751Buketlabpos3',Strokedata)



