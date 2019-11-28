
## Reset workspace (optional)
# print('Resetting python workspace ...')
# from IPython import get_ipython
# get_ipython().magic('reset -sf')

## Set the current working directory
import sys, os
import numpy as np
import torch
import hdf5storage
import torch.nn.functional as F
# os.chdir('/scratch2/chowdh51/Code/TIFS2019/Deployment')
#
# print('Setting current working directory to ...')
# print(os.getcwd())


from models import OneD_Triplet_CNN as network

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


## Load a sample pre-trained 1D-Triplet-CNN(MFCC-LPC) model
model_save_path = 'trained_models/oned_triplet_cnn_best.pth.tar'
model = network.cnn().cuda()
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(model_save_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

## Set Cosine similarity as criterion
criterion_loss = F.cosine_similarity

## Load sample MFCC-LPC feature patches for testing
feature_path_1 = '/scratch2/chowdh51/Code/TIFS2019/Deployment/sample_feature/1055/1.mat'
mat = hdf5storage.loadmat(feature_path_1)
feature_1 = np.array(mat['data'])

feature_path_2 = '/scratch2/chowdh51/Code/TIFS2019/Deployment/sample_feature/1055/22.mat'
mat = hdf5storage.loadmat(feature_path_2)
feature_2 = np.array(mat['data'])

feature_path_3 = '/scratch2/chowdh51/Code/TIFS2019/Deployment/sample_feature/1066/12.mat'
mat = hdf5storage.loadmat(feature_path_3)
feature_3 = np.array(mat['data'])

feature_path_4 = '/scratch2/chowdh51/Code/TIFS2019/Deployment/sample_feature/1066/17.mat'
mat = hdf5storage.loadmat(feature_path_4)
feature_4 = np.array(mat['data'])


## Compare the genuine audio sample pair represented by feature_1 and feature_2
data_1 = torch.from_numpy(np.expand_dims(feature_1.transpose((2, 0, 1)),axis=0)).float()
data_2 = torch.from_numpy(np.expand_dims(feature_2.transpose((2, 0, 1)),axis=0)).float()
embedding_1 = model(data_1)
embedding_2 = model(data_2)
dist_val_g = criterion_loss(embedding_1, embedding_2)
print(('Genuine Match Score = {0}').format(dist_val_g.data.cpu().numpy()[0]))

## Compare the genuine audio sample pair represented by feature_3 and feature_4
data_1 = torch.from_numpy(np.expand_dims(feature_3.transpose((2, 0, 1)),axis=0)).float()
data_2 = torch.from_numpy(np.expand_dims(feature_4.transpose((2, 0, 1)),axis=0)).float()
embedding_1 = model(data_1)
embedding_2 = model(data_2)
dist_val_g = criterion_loss(embedding_1, embedding_2)
print(('Genuine Match Score = {0}').format(dist_val_g.data.cpu().numpy()[0]))

## Compare the impostor audio sample pair represented by feature_1 and feature_3
data_1 = torch.from_numpy(np.expand_dims(feature_1.transpose((2, 0, 1)),axis=0)).float()
data_2 = torch.from_numpy(np.expand_dims(feature_3.transpose((2, 0, 1)),axis=0)).float()
embedding_1 = model(data_1)
embedding_2 = model(data_2)
dist_val_g = criterion_loss(embedding_1, embedding_2)
print(('Impostor Match Score = {0}').format(dist_val_g.data.cpu().numpy()[0]))

## Compare the impostor audio sample pair represented by feature_2 and feature_4
data_1 = torch.from_numpy(np.expand_dims(feature_2.transpose((2, 0, 1)),axis=0)).float()
data_2 = torch.from_numpy(np.expand_dims(feature_4.transpose((2, 0, 1)),axis=0)).float()
embedding_1 = model(data_1)
embedding_2 = model(data_2)
dist_val_g = criterion_loss(embedding_1, embedding_2)
print(('Impostor Match Score = {0}').format(dist_val_g.data.cpu().numpy()[0]))
