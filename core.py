import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import numpy as np
# curvatubes imports
from cvtub.utils import slices, single, load_nii, save_nii, random_init, init_balls
from cvtub.energy import discrepancy, ratio_discr
from cvtub.generator import _generate_shape
from cvtub.curvdiags import kap_eps, curvhist, density_scatter
import torch
from os.path import dirname, join as pjoin
import scipy.io as sio


# Main function that generates 3D shapes with curvatubes
# optimizer: Adam
# flow type: conservative H^{-1}
# periodic boundary conditions

def generate(A0, params, M0, delta_x, maxeval = 10000,
            #snapshot_folder = '', exp_title = '',
             display_all = True, check_viable = False, cond_take_snapshot = None) :

    '''Optimizes the phase-field Feps(u) ( see paper / see comments in cvtub/energy.py ) '''

    xi =  1e-6
    flow_type = 'cons'
    mode = 'periodic'
    optim_method = 'adam'
    sigma_blur = 2
    Z,X,Y = A0[0].shape

    optim_props = {'maxeval': maxeval, 'sigma_blur': sigma_blur, 'lr': .001, 'eps_adam' : 1e-2,
                   'betas' : (0.9,0.999), 'weight_decay' : 0, 'amsgrad' : False,
                   'display_it_nb' : 1000, 'fill_curve_nb' : 50}

    u = _generate_shape(A0, params, delta_x, xi, optim_method, optim_props, flow_type, mode,
                               M0 = M0,
                               #snapshot_folder = snapshot_folder, exp_title = exp_title,
                                cond_take_snapshot = cond_take_snapshot, display_all = display_all,
                                check_viable = check_viable)

    if check_viable == True :
        u, viable_bool, E = u
        return u.detach().cpu().numpy(), viable_bool, E

    return u.detach().cpu().numpy(), None


# Function which plots the curvature diagram of a shape
# defined as the zero level set of a phase-field u
def plot_curvature_diagram(u, iter_i, mat, delta_x, params,  bin_num, save = True, save_name = 'curvature_diagram.png'):

    kap1_eps, kap2_eps = kap_eps(u)
    kap1_vals, kap2_vals, areas, total_area, total_volume, w3, w4, cube = curvhist(u, iter_i, mat, kap1_eps, kap2_eps, delta_x = delta_x, show_figs = False)
    x,y = np.clip(kap1_vals, -100,100), np.clip(kap2_vals, -100, 100)
    data = density_scatter(x,y, areas, params, showid = True, equalaxis = True,
                    bins = bin_num, xlabel = 'kap1',
                    ylabel = 'kap2', save = save, save_name = save_name)
    kap_vals = np.vstack((kap1_vals,kap2_vals)).T

    return data, cube, kap_vals, areas, total_area, total_volume, w3, w4


# nonlinear normalization function for curvature maps
class curvature_normalization:
    def __init__(self):
        self.a = 0.0007868399976657144
        self.b = 7.1482721674687655
        self.c = 0.9998899691305907
        self.d = 0.00078684
        self.e = 3.57414
        self.f = 0.00078684

    def normalize(self, data):
        data_nor = np.log(data+self.a)/self.b + self.c
        return data_nor

    def unnormalize(self, data):
        data_unnor = self.d * np.exp(self.e * data) - self.f
        return data_unnor


# normalization functions for design parameters
class design_para_normalization:
    def __init__(self,data):
        self.mu = torch.mean(data,dim=0)
        self.std = torch.std(data,dim=0)
        self.min = torch.min(data,dim=0)[0]
        self.max = torch.max(data,dim=0)[0]
        self.diff = self.max - self.min
        self.cols = data.size()[1]

    def normalize(self, data):
        data = data.clone()
        for i in range(0, self.cols):
            data[:,i] = torch.div(data[:,i]-self.min[i], self.max[i]-self.min[i])
        return data

    def unnormalize(self, data):
        data = data.clone()
        for i in range(0, self.cols):
            data[:,i] = torch.mul(data[:,i], self.max[i]-self.min[i]) +self.min[i]
        return data


# convert cuvature vector to curvature map
def from_20100_to_200(y):

    length = len(y)
    y_long = np.zeros([length,40000])
    count = -1
    try: 
        id_save = np.load(os.getcwd() +'/data/id_save.npy')
    except:
        id_save = np.load(os.path.dirname(module_path) +'/data/id_save.npy')
    for i in range(40000):
        if i in id_save:
            count += 1
            y_long[:,i] = y.detach().cpu().numpy()[:,int(count)]
    y_matrix = y_long.reshape(length,200,200)

    return y_matrix


# convert cuvature map to curvature vector
def from_200_200_to_20100(data):

    x_reduce_list = []
    try: 
        id_del = np.load(os.getcwd() + '/data/id_del.npy')
    except:
        id_del = np.load(os.path.dirname(module_path) + '/data/id_del.npy')

    for iter_i in range(len(data)):
        data_reshape = data[iter_i].reshape(1,40000)
        data_reduce = np.delete(data_reshape,id_del,1)#(1,20100)
        if np.isnan(data_reduce.max()):
            print(iter_i)
        x_reduce_list.append(data_reduce)

    return np.array(x_reduce_list).reshape(-1,20100)


# forward model
class forward(torch.nn.Module):
    def __init__(self):
        super(forward, self).__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(7,50),
            torch.nn.ReLU(),
            torch.nn.Linear(50,150),
            torch.nn.ReLU(),
            torch.nn.Linear(150,300),
            torch.nn.ReLU(),
            torch.nn.Linear(300,600),
            torch.nn.ReLU(),
            torch.nn.Linear(600,1200),
            torch.nn.ReLU(),
            torch.nn.Linear(1200,2500),
            torch.nn.ReLU(),
            torch.nn.Linear(2500,5000),
            torch.nn.ReLU(),
            torch.nn.Linear(5000,10000),
            torch.nn.ReLU(),
            torch.nn.Linear(10000,20000),
            torch.nn.ReLU(),
            torch.nn.Linear(20000,20100),
            torch.nn.ReLU6()
        )
    def forward(self, x):
        encode = self.linear(x)/6
        return encode


# inverse model
class inverse(torch.nn.Module):
    def __init__(self):
        super(inverse, self).__init__()
        self.fc1 = torch.nn.Linear(20100,5000)
        self.fc2 = torch.nn.Linear(5000,1000)
        self.fc3 = torch.nn.Linear(1000,200)
        self.fc4 = torch.nn.Linear(200,40)
        self.fc5 = torch.nn.Linear(40,7)
        self.linear = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            self.fc2,
            torch.nn.ReLU(),
            self.fc3,
            torch.nn.ReLU(),
            self.fc4,
            torch.nn.ReLU(),
            self.fc5,
            torch.nn.ReLU6()
        )

    def forward(self, x):
        encode = self.linear(x)/6
        return encode
    


# data preparation for inverse design benchmarks
def data_prepare(u, output_dir):

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # can be any value just for calling curvature plot function
    a20 = np.random.uniform(0,1)                
    a11 = np.random.uniform(-2,2)
    a02 = np.random.uniform(0,1)
    b10 = np.random.uniform(-200,200)
    b01 = np.random.uniform(-200,200)
    c = np.random.uniform(-5000,5000)
    M0 = np.random.uniform(-0.8,-0.15)            
    E = torch.Tensor(np.array(1.0)).type(dtype)

    os.makedirs(output_dir, exist_ok = True)
    bin_num = np.linspace(-100,100,201)
    mat = True
    data, cube, kap_vals, areas, total_area, total_volume, w3, w4 = plot_curvature_diagram(u, 0, mat, 0.01,(0.01, a20, a11, a02, b10, b01, c, E), bin_num, save = True, save_name = output_dir + '/kap1_kap2.png')
    np.save(output_dir+'/u.npy',u)
    np.save(output_dir+'/kap_vals.npy',kap_vals)
    np.save(output_dir+'/data.npy',data)
    cube.save(output_dir+'/mesh.stl')


# inverse design for a target curvature distribution
def inverse_design(y_test, cuda_device):

    from pickle5 import load
    from collections import OrderedDict
    #load inverse model

    try: 
        inverse_path = os.getcwd() + '/model/trained_model/inverse.pt'
        data_scaler = load(open( os.getcwd() + '/data/scaler.pkl', 'rb')) # normalization factor
    except:
        inverse_path = os.path.dirname(module_path) + '/model/trained_model/inverse.pt'
        data_scaler = load(open(os.path.dirname(module_path)+'/data/scaler.pkl', 'rb')) # normalization factor
    inverse_model = inverse()

    state_dict = torch.load(inverse_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    inverse_model.load_state_dict(new_state_dict)
    inverse_model.to(cuda_device)

    # inverse design parameters
    x_pred = inverse_model(y_test)#7,normalized
    x_pred_unnormalize = data_scaler.unnormalize(x_pred)#7,unnormalize
    x_pred_unnormalize = x_pred_unnormalize.detach().cpu().numpy()

    return x_pred_unnormalize
