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


# normalization functions for design parameters
class Normalization:
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
