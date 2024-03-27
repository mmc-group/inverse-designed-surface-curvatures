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
import math
from sympy import *
from scipy import special

cuda_device = 0
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# ============================data prepare======================================================
# load data
path = module_path +'/spinodoid/GRF_file/GRF.mat'
GRF_contents = sio.loadmat(path)
u = GRF_contents[sorted(GRF_contents.keys())[0]]
# m = -0.4
density = 0.3 # density = (1+m)/2
lev = special.erfinv(2*density-1)*math.sqrt(2) # isosurface level set
u = np.tanh((-u+lev)) # fit the tanh profile near the interface 

# output curvature plot by calling the curvature plot function
output_dir = module_path +'/spinodoid/data_a-1/'
data_prepare(u, output_dir)

# ============================inverse design======================================================
# load data
data_arrayset = np.load(module_path + '/spinodoid/data_a-1/data.npy', allow_pickle=True).T.reshape(-1,200,200)
data_arrayset = curvature_normalization().normalize(data_arrayset)
x_arrayset = from_200_200_to_20100(data_arrayset)
y_test = torch.tensor(x_arrayset).type(dtype)#20100

# inverse design
x_test_pred_unnormalize = inverse_design(y_test, cuda_device)

print('inverse designed parameters:', x_test_pred_unnormalize)

torch.cuda.empty_cache()