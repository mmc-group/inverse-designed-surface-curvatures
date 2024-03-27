#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 9 11:37:02 2020

@author: Anna SONG

Computes the curvature diagrams of a surface {u = 0} defined by a phase-field u
with profile close to a tanh profile. Curvature diagrams are histograms of the
diffuse principal curvatures (kap_{1,eps}, kap_{2,eps}) on the surface, computed
with respect to a triangular mesh of the surface.

The statistics of these curvatures characterize the *texture* of a 3D shape.

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn

import skimage
from scipy.interpolate import RegularGridInterpolator as RGI

import torch
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
from cvtub.utils import double_W_prime, manual_softplus
from cvtub.energy import auxiliary_function
#from cvtub.generator import _generate_shape
import os.path as osp
import unittest
from matplotlib.image import NonUniformImage
from scipy.interpolate import interpn
from matplotlib import cm
from matplotlib import cbook
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import dirname, join as pjoin
import scipy.io as sio
from stl import mesh
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['text.usetex'] = True
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
# np.set_printoptions(precision=3, suppress=True)

def kap_eps(u, eps = 0.02, delta_x = 0.01, mode = 'periodic', xi = 1e-6) :

    if type(u) != torch.Tensor :
        u = torch.tensor(u).type(dtype)
    grad, Hess_diag, Hess_off = auxiliary_function(u, eps, delta_x, mode)

    dz_u = grad[...,0] ;      dx_u = grad[...,1];       dy_u = grad[...,2]
    Hzz_u = Hess_diag[...,0]; Hxx_u = Hess_diag[...,1]; Hyy_u = Hess_diag[...,2]
    Hzx_u = Hess_off[...,0] ; Hzy_u = Hess_off[...,1] ; Hxy_u = Hess_off[...,2]

    norm_grad_sq = (grad ** 2).sum(-1)
    ngd_u = (norm_grad_sq + xi**2).sqrt()
    Wprim = double_W_prime(u)

    # nHn is n^T H n where H is the hessian of u and n approximates the direction
    # of the gradient (not well defined when grad u ~ 0) up to a small term xi
    nHn = Hzz_u * dz_u**2 + Hxx_u * dx_u**2 + Hyy_u * dy_u**2 \
        + 2 * (Hzx_u * dz_u * dx_u + Hzy_u * dz_u * dy_u + Hxy_u * dx_u * dy_u)
    nHn /= (norm_grad_sq + xi**2)
    # print(type(eps),type(Hzz_u),type(Wprim))
    Tra = - eps * (Hzz_u + Hxx_u + Hyy_u) + Wprim / eps

    Nor_pow2 = (eps**2) * ( (Hess_diag**2).sum(-1) + 2 * (Hess_off**2).sum(-1) ) \
             + Wprim**2 / (eps**2) - 2 * Wprim * nHn

    positive_part = manual_softplus(2 * Nor_pow2 - Tra**2)
    sqrt_part = torch.sqrt(positive_part)

    #Heps = Tra / (eps * ngd_u)
    #Keps = (Tra**2 - Nor_pow2) / (2 * eps**2 * (norm_grad_sq + xi**2) )
    kap1_eps = (Tra + sqrt_part) / (2 * eps * ngd_u)
    kap2_eps = (Tra - sqrt_part) / (2 * eps * ngd_u)
    #print('kap1_eps * kap2_eps may be different from Keps')

    kap1_eps = kap1_eps.detach().cpu().numpy()
    kap2_eps = kap2_eps.detach().cpu().numpy()

    return kap1_eps, kap2_eps


def curvhist(vol, iter_i, mat, kap1_eps, kap2_eps, lev = 0.0, delta_x = 0.01,
             show_figs = True, bins = 100,  save = True, save_name = 'default.png') :

    Z,X,Y = vol.shape
    print('Analysis for level set u = {}'.format(lev))
    if mat == True:
    #     path = '/home/yaqi/curvature/matfile/'
    #     F_contents = sio.loadmat(path+'F{}.mat'.format(iter_i))
    #     V_contents = sio.loadmat(path+'V{}.mat'.format(iter_i))
    #     # GRF_contents = sio.loadmat(path+'GRF.mat')
    #     faces = F_contents['F{}'.format(iter_i)]-1
    #     verts = V_contents['V{}'.format(iter_i)]
    #     # vol = GRF_contents['GRF']
    #     # verts = verts*100
        # verts, faces, normals, values_mc = skimage.measure.marching_cubes_lewiner(vol, level = lev)
        verts, faces, normals, values_mc = skimage.measure.marching_cubes(vol, level = lev)
        # Create the mesh
        cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                cube.vectors[i][j] = verts[f[j],:]
        # cube.save('/home/yaqi/curvature/bone_mesh/mesh-{}.stl'.format(iter_i))
        # cube.save('/home/yaqi/curvature/matfile/mesh-{}.stl'.format(iter_i))
    else:
        # verts, faces, normals, values_mc = skimage.measure.marching_cubes_lewiner(vol, level = lev)
        verts, faces, normals, values_mc = skimage.measure.marching_cubes(vol, level = lev)
        cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                cube.vectors[i][j] = verts[f[j],:]
    # print(save_name)

    #compute cells areas
    As = verts[faces[:,0]]
    Bs = verts[faces[:,1]]
    Cs = verts[faces[:,2]]
    Gs = (As + Bs + Cs) / 3 # barycenters of the mesh cells
    alen = np.linalg.norm(Cs - Bs, axis = 1)
    blen = np.linalg.norm(Cs - As, axis = 1)
    clen = np.linalg.norm(Bs - As, axis = 1)
    midlen = .5 * (alen + blen + clen)
    areas = np.sqrt(midlen * (midlen - alen) * (midlen - blen) * (midlen - clen))
    total_area = areas.sum()
    print('Total area of the mesh is ',total_area * delta_x**2, ' (real unit)')
    total_volume = (vol > lev).sum()
    print('Total volume of the solid: ', total_volume * delta_x**3, 'approximately (real unit),')
    print('i.e. ', total_volume / (Z*X*Y)*100, '% of the total domain volume. \n')
    # interpolate the values of kap1_eps and kap2_eps on the barycenters of each mesh cell
    aux_grid = (np.arange(Z),np.arange(X),np.arange(Y))

    interpolator_kap1 = RGI(aux_grid, kap1_eps, method = 'linear')
    kap1_vals = interpolator_kap1(Gs)

    interpolator_kap2 = RGI(aux_grid, kap2_eps, method = 'linear')
    kap2_vals = interpolator_kap2(Gs)
    w3 = ((kap1_vals + kap2_vals)*areas*delta_x**2/2).sum()
    w4 = (kap1_vals*kap2_vals*areas*delta_x**2).sum()

    if show_figs :
        'plot histogram bins --- probability densities'

        fig , ax = plt.subplots(2,2,figsize = (12,12))
        heights_kap1, bins_kap1,_ = ax[1,0].hist(kap1_vals, bins=bins, density=True, weights=areas)
        ax[1,0].set_title('kap1', y = -0.15, fontsize = 20)
        heights_kap2, bins_kap2,_ = ax[1,1].hist(kap2_vals, bins=bins, density=True, weights=areas)
        ax[1,1].set_title('kap2', y = -0.15, fontsize = 20)

        if save :
            plt.savefig(save_name)
        # plt.show()

    return kap1_vals, kap2_vals, areas, total_area, total_volume, w3, w4, cube

def density_scatter(x, y, areas, params, xlabel = '', ylabel = '', showid = False, showparab = False,
                    equalaxis = False, size = 0.1, bins = 10, sort = True,
                    showfig = True, save = False, save_name = 'histogram.png', **kwargs )   :
    'Plots a beautiful colored scatter plot of a 2D histogram (such as curvature diagrams)'
    # this code comes from StackOverflow
    eps, a20, a11, a02, b10, b01, c, E = params
    fig , ax = plt.subplots(figsize = (11,10))
    data, x_e, y_e = np.histogram2d(x, y, weights = areas, bins = bins, density = True)
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)
    # Sort the points by density, so that the densest points are plotted last
    sort=False
    # if sort :#asend
    #     idx = z.argsort()
    #     x, y, z = x[idx], y[idx], z[idx]
    if showid :
        # mini = min(x.min(), y.min())
        # maxi = max(x.max(), y.max())
        ax.plot([-100, 100], [-100, 100], color = 'red', linewidth = 2)
    # if showparab :
    #     T = np.linspace(x.min(),x.max(), 100)
    #     ax.plot(T,(T**2)/4, color = 'red', linewidth = 2)

    im = ax.scatter( x, y, c=z, s = size)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cax.yaxis.set_tick_params(labelsize=20)
    cbar=plt.colorbar(im, cax=cax)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel(r"$\kappa_1$", fontsize=20)
    ax.set_ylabel(r"$\kappa_2$", fontsize=20)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.autoscale_view()
    ##############################
    # delta=0.25#step
    # xx = np.arange(-100, 100,delta)
    # yy = np.arange(-100, 100,delta)
    # X,Y = np.meshgrid(xx,yy)
    # Z = a20*X**2+a11*X*Y+a02*Y**2+b10*X+b01*Y+c
    # if E.item()<0:
    #     contour1=ax.contour(X,Y,Z,[E.item(),0],colors=('m','g'))
    #     ax.clabel(contour1,fontsize=10,colors=('m','g'))
    # else:
    #     contour1=ax.contour(X,Y,Z,[0,E.item()],colors=('g','r'))
    #     ax.clabel(contour1,fontsize=10,colors=('g','r'))
############################################
    # norm = Normalize(vmin = np.min(z), vmax = np.max(z))
#     cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    # if save :
    plt.savefig(save_name)
    # if showfig :
        # plt.show()
    # else :
        # plt.close()
#----------------------plot histogram and save output--------------------------
    fig , ax = plt.subplots(figsize = (11,8))
    im = plt.imshow(data.T, interpolation='nearest', origin='lower',extent=[x_e[0], x_e[-1], y_e[0], y_e[-1]])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.yaxis.set_tick_params(labelsize=20)
    cbar=plt.colorbar(im, cax=cax)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.autoscale_view()
    plt.savefig(save_name[:-12]+'.png')
#----------------------save dataset--------------------
    return data
###############################################################################################################
    # fig, ax = plt.subplots(1, 4, figsize=(32, 6))
    # im = ax[0].imshow(data.T, interpolation='nearest', origin='lower',extent=[x_e[0], x_e[-1], y_e[0], y_e[-1]])
    # divider = make_axes_locatable(ax[0]) # create an axes on the right side of ax. The width of cax will be 5% # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # cax = divider.append_axes("right", size="5%", pad=0.2)
    # plt.colorbar(im, cax=cax)
    # #------------------------------first-----------------------------------
    # print('x_e',x_e[0],x_e[-1])
    # k1, k2 = np.meshgrid(0.5*(x_e[1:] + x_e[:-1]),0.5*(y_e[1:]+y_e[:-1]))
    # cloud = np.vstack([k1.reshape(1,-1),k2.reshape(1,-1),data.T.reshape(1,-1)]).T
    # data_id = np.where(cloud[:,2] < lower_lim)
    # cloud_main = np.delete(cloud, data_id[0],0)
    # print(cloud_main.shape)
    # ax[1].scatter(cloud_main[:,0],cloud_main[:,1],c=cloud_main[:,2], s = 1)
    # ax[1].set_xlim(-100, 100)
    # ax[1].set_ylim(-100, 100)
    # norm = Normalize(vmin = cloud_main[:,2].min(), vmax = cloud_main[:,2].max())
    # fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax[1])
    # #------------------------------second-----------------------------------
    # z = interpn(( 0.5*(x_e[1:] + x_e[:-1]),0.5*(y_e[1:]+y_e[:-1])),data , np.vstack([x,y]).T,method = "splinef2d", bounds_error = False)
    # ax[2].scatter(x,y,c=z, s = 1)
    # ax[2].set_xlim(-100, 100)
    # ax[2].set_ylim(-100, 100)
    # nan_id = np.argwhere(np.isnan(z))
    # z_del = np.delete(z, nan_id, 0)
    # norm = Normalize(vmin = z_del.min(), vmax = z_del.max())
    # fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax[2])
    # #------------------------------third-----------------------------------
    # points = np.vstack([x,y]).T
    # points_del = np.delete(points, nan_id, 0)
    # zero_id = np.where(z_del < lower_lim)
    # points_main = np.delete(points_del, zero_id[0],0)
    # #------------------------------forth-----------------------------------
    # z_main = interpn(( 0.5*(x_e[1:] + x_e[:-1]),0.5*(y_e[1:]+y_e[:-1])),data,points_main,method = "splinef2d", bounds_error = False)
    # ax[3].scatter(points_main[:,0],points_main[:,1],c=z_main, s = 1)
    # ax[3].set_xlim(-100, 100)
    # ax[3].set_ylim(-100, 100)
    # #norm = Normalize(vmin = z_main.min(), vmax = z_main.max())
    # fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax[3])
    # print(data.shape,cloud_main.shape,z.shape,points_main.shape)
    # plt.savefig(save_name[:-10]+'.png')
