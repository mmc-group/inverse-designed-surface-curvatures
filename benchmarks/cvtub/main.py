import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
parent = os.path.dirname(module_path)
sys.path.append(parent)
import torch
import numpy as np
from core import *

cuda_device = 0
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# load data
data_arrayset = np.load(module_path+'/cvtub/data_a-1/data.npy', allow_pickle=True).T.reshape(-1,200,200)
data_arrayset = curvature_normalization().normalize(data_arrayset)
x_arrayset = from_200_200_to_20100(data_arrayset)
y_test = torch.tensor(x_arrayset).type(dtype)#20100

# inverse design
x_test_pred_unnormalize = inverse_design(y_test, cuda_device)

print('inverse designed parameters:', x_test_pred_unnormalize)

