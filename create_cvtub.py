import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import torch
import numpy as np
from core import *
from cvtub.energy import ratio_discr


dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

foldername = '/benchmarks/cvtub/'

# resolution of generated structures
resolution = 100
eps = 0.02
Z,X,Y = resolution,resolution,resolution
delta_x = 1/resolution
A0 = 40 * delta_x * np.random.rand(3,Z,X,Y) # random starting phase field
A0 = torch.Tensor(A0).type(dtype)

file_num = 0
num_samples = 1
mat = False # flag for inverse design data source 
for iter_i in range(1,num_samples+1):

    print('\n\n---------- {}/{} ----------'.format(iter_i,num_samples))
    print('file_num-{}'.format(file_num))
    MAXEVAL = 1000

    # input parameters
    a20 = 1                 #------1
    a11 = 0
    a02 = 10
    b10 = -100
    b01 = 200
    c = 3500
    M0 = -0.4              #------7
    params_a = np.array([a20,a11,a02,b10,b01,c, M0])

    #================ BEGIN HERE ===============
    u, viable_bool, E = generate(A0, (eps, a20, a11, a02, b10, b01, c), M0,
        delta_x, maxeval = MAXEVAL,
        check_viable = True, display_all = False)
    
    if viable_bool:

        file_num += 1
        print('**********',file_num,'**********')
        output_name = 'data_a-{}'.format(file_num)
        output_dir = os.getcwd() + foldername + output_name
        if output_dir[-1] != '/':
            output_dir += '/'
        os.makedirs(output_dir, exist_ok=True)

        ratio_discrepancy = ratio_discr(u)
        bin_num = np.linspace(-100,100,201)
        data, cube, kap_vals, areas, total_area, total_volume, w3, w4 = plot_curvature_diagram(u, iter_i, mat, delta_x,(eps, a20, a11, a02, b10, b01, c, E), bin_num, save = True, save_name = output_dir + 'kap1_kap2.png')

        np.save(output_dir+'/u.npy',u)
        np.save(output_dir+'/kap_vals.npy',kap_vals)
        np.save(output_dir+'/data.npy',data)
        np.save(output_dir+'/params_a.npy',params_a)
        cube.save(output_dir+'/mesh.stl')

    else:
        pass

torch.cuda.empty_cache()
