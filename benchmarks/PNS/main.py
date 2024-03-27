import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
parent = os.path.dirname(module_path)
sys.path.append(parent)
from core import *
import torch
import math
import numpy as np

cuda_device = 0
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# ============================data preparation======================================================
# periodic surface function
X, Y, Z = np.mgrid[-2*math.pi:2*math.pi:100j, -2*math.pi:2*math.pi:100j, -2*math.pi:2*math.pi:100j]
f1 = 0.6
f2 = 0.3
iter_i = 1
u = (np.cos(X*iter_i)*np.sin(f1*Y*iter_i)*np.sin(f2*Z*iter_i)
        + np.cos(Y*iter_i)*np.sin(f1*Z*iter_i)*np.sin(f2*X*iter_i)
        + np.cos(Z*iter_i)*np.sin(f1*X*iter_i)*np.sin(f2*Y*iter_i))
u = u-0.3 # level set at 0.3
u = np.tanh(u) # fit tanh profile

output_dir = module_path + '/PNS/data_a-1'
data_prepare(u, output_dir)

# ============================inverse design======================================================
# load data
data_arrayset = np.load(module_path + '/PNS/data_a-1/data.npy', allow_pickle=True).T.reshape(-1,200,200)
data_arrayset = curvature_normalization().normalize(data_arrayset)
x_arrayset = from_200_200_to_20100(data_arrayset)
y_test = torch.tensor(x_arrayset).type(dtype) #20100

# inverse design
x_test_pred_unnormalize = inverse_design(y_test, cuda_device)

print('inverse designed parameters:', x_test_pred_unnormalize)

torch.cuda.empty_cache()
