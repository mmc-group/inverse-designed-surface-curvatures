import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
parent = os.path.dirname(module_path)
sys.path.append(parent)
from core import *
import torch
import numpy as np


cuda_device = 0
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
torch.cuda.set_device(cuda_device)

# ============================data preparation======================================================
# load data
# define 2 CNN layers as smoothing function
weight1 = (torch.zeros(4, 4, 4)+0.1).view(1, 1, 4, 4, 4)
weight2 = (torch.zeros(3, 3, 3)+0.1).view(1, 1, 3, 3, 3)
m = torch.nn.ConvTranspose3d(1,1,(4,4,4),stride=(2,2,2),padding =(1,1,1),bias=False)
n = torch.nn.Conv3d(1,1,(3,3,3),stride=(2,2,2),padding =(1,1,1),bias=False)
with torch.no_grad():
    m.weight = torch.nn.Parameter(weight1)
    n.weight = torch.nn.Parameter(weight2)  

#============================
GRF_contents = sio.loadmat(module_path + '/bone/scan_CT_file/scan_1.mat')
u0 = GRF_contents['f']
u1 = (u0-63.5)/63.5 # rescale the field value to [-1,1]
input_tensor = torch.tensor(u1).view(1,1,100,100,100).float() 
output=n(m(n(m(input_tensor)/0.8)/2.7)/0.8)/2.7 # smooth the field value by 3D CNN layers, which averages the field values within the kernel
u = output.detach().numpy().reshape(100,100,100)
u = np.tanh(u) # fit the tanh profile near the interface 

output_dir = module_path + '/bone/data_a-1'
data_prepare(u, output_dir)

# ============================inverse design======================================================
# load data
data_arrayset = np.load(module_path + '/bone/data_a-1/data.npy', allow_pickle=True).T.reshape(-1,200,200)
data_arrayset = curvature_normalization().normalize(data_arrayset)
x_arrayset = from_200_200_to_20100(data_arrayset)
y_test = torch.tensor(x_arrayset).type(dtype) #20100

# inverse design
x_test_pred_unnormalize = inverse_design(y_test, cuda_device)

print('inverse designed parameters:', x_test_pred_unnormalize)

torch.cuda.empty_cache()

