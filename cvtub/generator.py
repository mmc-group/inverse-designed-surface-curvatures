#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 7 14:35:20 2020

@author: Anna SONG

Main wrapper function for generating 3D tubular and membranous shape textures.

See my article at       https://arxiv.org/abs/2103.04856


Note: convention that u has shape (Z,X,Y)

"""

import time
import numpy as np
import matplotlib.pyplot as plt

import torch

from cvtub.utils import save_nii, slices, project_average
from cvtub.filters import GeneralGaussianBlur3D_periodic, GeneralGaussianBlur3D_notperiodic
from cvtub.filters import my_custom_GradHess

from cvtub.energy import polykap_deg2, ratio_discr


def _generate_shape(v0, params, delta_x, xi, optim_method, optim_props,
                    flow_type, mode, M0 = None,
                   #snapshot_folder = '',
                   exp_title = '',
                   cond_take_snapshot = None, display_all = True,
                   return_var = False, return_energy = False, check_viable = False) :

    ''' Optimizes the phase-field energy

            kap_poly_deg2(u) = Feps(u)

        with the parameters

            params = eps, a20, a11, a02, b10, b01, c, [mu, theta]    (optional)

        (floats for constants, torch.tensor for space-dependent values)

        and the flow_type:
            - 'L2':     \dot{u} = - dE/du
                standard L2 gradient flow
                working with u
            - 'averm0': \dot{u} = - Pi_0( dE/du )   where Pi_0(v) = v - bar(v) cancels the average
                average-projected flow, globally mass-preserving
                working with u
            - 'cons':   \dot{u} = Laplacian( dE/du )
                conservative H^{-1} flow, locally mass-preserving
                working with a vector field A and setting u = div A + M0
            # averm0 not shown in the paper

        mode specifies the boundary conditions:
            - 'periodic': u has the same values on opposite faces of the cubic domain
            - 'replicate': imposes {grad u perp to n} where n is the normal
                    at the border of the domain

        optim_method: 'adam' or 'bfgs'
        optim_props: internal parameters of Adam or BFGS
            note: I also included the sigma_blur of the Gaussian kernel there

        v0: initialization
            = u0 if flow_type = 'L2', 'averm0'
            = A0 if flow_type = 'cons', with change of variable u = div A + M0

        M0: imposed mass (i.e. average) of u, between -1 and 1
            if flow_type = 'averm0' or 'cons'
            not used if flow_type = 'L2'

        delta_x: mathematical length represented by 1 pixel
        xi: 1e-6 typically, to regularize |grad u| and 1 / |grad u|^2


        # Other options:
        return_var: if True, return u and the variable uu or A (considered by pytorch)
            note: u = k \ast uu
               or u = k \ast (div A + M0)
               where k is a small Gaussian blur,
               although by a small abuse of notation we omit k
        cond_take_snapshot(iteration) is True when it is time to take a photo of u

    '''

    global iteration, n_evals, first_snapshot, title, nan_OK, viable_OK, u, uu
    global E_curve, grad_L1mean_curve, grad_max_curve, M_curve
    global params2, M02 # a copy of params and M0 to avoid an error I cannot understand

    if len(params) == 9 :
        eps, a20, a11, a02, b10, b01, c, mu, theta = params
        orientation = True
    elif len(params) == 11 :
        eps, a20, a11, a02, b10, b01, c, mu, theta, mu_2, theta_2 = params
        orientation = True
    elif len(params) == 13 :
        eps, a20, a11, a02, b10, b01, c, mu, theta, mu_2, theta_2, mu_3, theta_3 = params
        orientation = True
    elif len(params) == 7 :
        eps, a20, a11, a02, b10, b01, c = params
        orientation = False
    else :
        raise ValueError("Params should be a tuple (eps, a20, a11, a02, b10, b01, c, [mu, theta]) (mu and theta optional)")

    if type(M0) != torch.Tensor and M0 is not None and (M0 < -1 or M0 > 1) :
        raise ValueError("M0 should be a number between -1 and 1 (we recommend a value between -0.7 and -0.2 for simulations)")
    if M0 is not None and flow_type == 'averm0' :
        raise ValueError("Set M0 = None because in flow_type = 'averm0', the mass is just the one of u0 at start")

    spatialized = False
    for x in params: # if one of the parameters is space-dependent then spatialized = True
        #if type(x) != int and type(x) != float:
        if type(x) == torch.Tensor :
            spatialized = True

    if flow_type in ['L2','averm0'] :
        Z,X,Y = v0.shape # scalar field u = v0
    elif flow_type == 'cons' :
        if mode == 'replicate' :
            raise ValueError("With flow_type = 'cons' only periodic border conditions are considered.")
        A0 = v0 # vector field A = v0
        Z,X,Y = A0[0].shape
    else :
        raise ValueError("flow_type should be one of 'L2', 'averm0', or 'cons'.")

    # Initializations...
    iteration = 0
    n_evals = 0
    next_disp = 0
    first_snapshot = True
    E_curve = []
    grad_L1mean_curve = []
    grad_max_curve = []
    M_curve = []
    nan_OK = True
    viable_OK = True

    maxeval = optim_props['maxeval']
    sigma_blur = optim_props['sigma_blur']
    display_it_nb = optim_props['display_it_nb']
    fill_curve_nb = optim_props['fill_curve_nb']

    #latex_mass = r'$\frac{1}{|\Omega|} \int_{{\Omega}} u(x) ~dx^3$'
    #latex_grad_L1mean = r'$\frac{1}{|\Omega|} \int_{{\Omega}} \left| \frac{\partial F}{\partial var} \right| ~dx^3 $'
    #latex_grad_max = r'$\max\limits_{{\Omega}} \left| \frac{\partial F}{\partial var} \right| $'
    #display(Latex(...))

    #if optim_method == 'sgd' : # others: dampening, nesterov
    #    learning_rate = optim_props['lr']
    #    momentum = optim_props['momentum']
    #    weight_decay = optim_props['weight_decay']

    if optim_method == 'adam' :
        betas_adam = optim_props['betas']
        learning_rate = optim_props['lr']
        eps_adam = optim_props['eps_adam'] # default = 1e-8
        weight_decay = optim_props['weight_decay']
        amsgrad = optim_props['amsgrad']

    elif optim_method == 'bfgs' :
        learning_rate = optim_props['lr']
        bfgs_max_iter = optim_props['bfgs_max_iter'] # along line search
        history_size = optim_props['history_size']
        line_search_fn = optim_props['line_search_fn']
    else :
        raise ValueError("optim_method should be one of 'adam' or 'bfgs'.")

    if mode == 'periodic' :
        gaussian_blur = GeneralGaussianBlur3D_periodic(Z,X,Y,sigma_blur,sigma_blur,sigma_blur).cuda()
    elif mode == 'replicate' :
        gaussian_blur = GeneralGaussianBlur3D_notperiodic(Z,X,Y,sigma_blur,sigma_blur,sigma_blur).cuda()
    else :
        raise ValueError("mode should be one of 'periodic' or 'replicate'.")

    LZ = Z * delta_x ; LX = X * delta_x ; LY = Y * delta_x # mathematical lengths of the domain

    title = 'polykap_deg2 '+ optim_method + ' ' + flow_type + ' ' + mode

    if not spatialized : # we write the values for constant parameters in the title
        def formatted(f): return format(f, '.3f').rstrip('0').rstrip('.')
        aux_tuple = [a20, a11, a02, b10, b01, c]
        aux_string = ' coeffs [{} {} {} {} {} {}]'
        if M0 is not None :
            aux_tuple += [M0]
            aux_string += ' m {}'
        str_tup = tuple([formatted(x) for x in aux_tuple])
        title += aux_string.format(*str_tup)

        if orientation and len(params) == 9:
            title += ' mu {} theta {}'.format(mu, theta)
        elif orientation and len(params) ==11:
            title += ' mu_1 {} theta_1 {} mu_2 {} theta_2 {}'.format(mu, theta, mu_2, theta_2)
        elif orientation and len(params) ==13:
            title += ' mu_1 {} theta_1 {} mu_2 {} theta_2 {} mu_3 {} theta_3 {}'.format(mu, theta, mu_2, theta_2, mu_3, theta_3)
    else :
        title += ' coeffs spatialized'

    print(title)
    print('dx = {:.3f} LZ = {:.2f} LX = {:.3f} LY = {:.3f} xi = {}'.format(delta_x,LZ,LX,LY,xi))
    print(optim_props)


    '''Lets go!! '''

    t1 = time.time()

    # Construct the differential operators Grad, Hess, Div with buffered memory
    GradHessian = my_custom_GradHess(mode)
    GradHessConv_ZXY = GradHessian(Z,X,Y).cuda() if torch.cuda.is_available() else GradHessian(Z,X,Y)
    #GradHessConv_ZXY = GradHessian(Z,X,Y)
    if flow_type == 'cons' :
        from cvtub.filters import my_custom_Div
        Divergence_class = my_custom_Div(mode)
        Divergence = Divergence_class(Z,X,Y).cuda() if torch.cuda.is_available() else Divergence_class(Z,X,Y)
        #Divergence = Divergence_class(Z,X,Y)
    # initialise u and uu
    if flow_type in ['L2','averm0'] :
        uu = v0.clone()
        uu.requires_grad = True
        u = gaussian_blur(v0) # this one is not really used by closure(), only for display

    elif flow_type == 'cons' :
        if M0 is None :
            M0 = 0
        uu = Divergence(A0) / delta_x + M0 # using + or - Div(A) actually does not matter
        u = gaussian_blur(uu)
        A = A0.clone()
        A.requires_grad = True
        #breakpoint()
    if flow_type == 'averm0' :
        M0 = u.mean()

    print('')
    print(" umin = {}, umax = {}".format(u.min().item(), u.max().item()) )
    print(" m =  {:.3f} \n".format(u.mean().item()) )

    # prepare the optimizers + PyTorch

    if flow_type in ['L2','averm0'] :
        variable = uu
    elif flow_type == 'cons' :
        variable = A
    #if optim_method == 'sgd' :
    #    optimizer = torch.optim.SGD([variable], lr = learning_rate, momentum = momentum)
    if optim_method == 'bfgs' :
        optimizer = torch.optim.LBFGS([variable], lr = learning_rate, max_iter = bfgs_max_iter, history_size = history_size, line_search_fn = line_search_fn)
    if optim_method == 'adam' :
        optimizer = torch.optim.Adam([variable], lr=learning_rate, betas=betas_adam, eps=eps_adam, weight_decay=weight_decay, amsgrad=amsgrad)

    params2 = params # create a copy because I don't know why there is a bug not finding the variable
    M02 = M0 # create a copy because I don't know why there is a bug not finding the variable

    def loss() :
        global u, uu, n_evals, iteration, params2, M02

        if flow_type == 'L2' :
            u = gaussian_blur(uu)
            E = polykap_deg2(u, params2, delta_x, xi, GradHessConv_ZXY)

        if flow_type == 'averm0' :
            u = gaussian_blur(uu)
            u_m0 = project_average(u, m = M02)
            E = polykap_deg2(u_m0, params2, delta_x, xi, GradHessConv_ZXY)

        if flow_type == 'cons' :
            uu = Divergence(A) / delta_x + M02
            u = gaussian_blur(uu)
            E = polykap_deg2(u, params2, delta_x, xi, GradHessConv_ZXY)

        #fidelity_term: may be used as future work in a Reg + Fid segmentation method
        #if fidelity_term is not None :
        #    Fid = fidelity_term(uu, ...)
        #else :
        #    Fid = 0

        if n_evals % display_it_nb == 0 :
            print('E = {}'.format(E.item()))

        n_evals += 1

        return E # Reg + Fid

    def track_E(E) :

        if np.isnan(E.item()) == True :
            global nan_OK
            nan_OK = False

        if n_evals % fill_curve_nb == 0 :
            global E_curve, Fid_curve, grad_L1mean_curve, grad_max_curve

            E_curve += [E.item()]

            grad_L1mean = (torch.sum(torch.abs(variable.grad)) / (Z * X * Y)).item()
            grad_L1mean_curve += [grad_L1mean]

            grad_max = (torch.abs(variable.grad).max()).item()
            grad_max_curve += [grad_max]

        if display_all and n_evals % 100 == 0 and len(E_curve) > 0 :
            print('{:=5d}: E = {:.2e}, '.format(n_evals, E_curve[-1]), end="")

        if display_all and (n_evals + 1) % 100 == 0:
            print("")

        if display_all and n_evals % display_it_nb == 0 and len(E_curve) > 0 :
            print('\n \n at eval {}'.format(n_evals))
            print('grad_L1mean = {:.4e}'.format(grad_L1mean))
            print('latex_grad_max = {:.4e}'.format(grad_max))

    def track_mass() :
        mass = u.mean().item()
        if n_evals % fill_curve_nb == 0 :
            global M_curve
            M_curve += [mass]

        if display_all and n_evals % display_it_nb == 0 and len(E_curve) > 0 :
                print('m =   {:.3f} \n'.format(mass))

    def take_snapshot() :
        global first_snapshot
        if cond_take_snapshot(iteration) and first_snapshot == True :
            print('\n \n TAKING A SNAPSHOT NOW \n \n')
            global u
            if optim_method == 'bfgs' :
                title_end = 'it_{}_eval_{}'.format(iteration, n_evals)
            else :
                title_end = 'it_{}'.format(iteration)
            #save_nii(u.detach().cpu().numpy(), snapshot_folder + title_end, compressed = True)
            first_snapshot = False

    def closure():
        if cond_take_snapshot is not None :
            take_snapshot()

        optimizer.zero_grad()
        E = loss()
        E.backward()

        track_E(E)
        track_mass()
        return E

    while nan_OK and viable_OK and n_evals <= maxeval :

        optimizer.step(closure)
        iteration += 1 # for sgd and adam, iteration = n_evals
        first_snapshot = True # retake snapshots for the next interesting iteration

        if check_viable and n_evals in [500,1000,5000] : # check 3 times
            rat = ratio_discr(u, eps = eps, delta_x = delta_x, mode = mode).item()
            good_max = .1 < u.max().item()
            good_min = u.min().item() < -.1
            if not(good_max and good_min and rat < .75) :
                viable_OK = False

        # we show -var.grad instead of +var.grad
        if display_all and optim_method == 'adam' and (n_evals in [100, 300] or n_evals % display_it_nb == 0) :
            print(" umin = {}, umax = {}".format(u.min().item(), u.max().item()) )
            slices(u, rescale = True)
            if flow_type != 'cons' :
                slices(-uu.grad, rescale = True)
        #if display_all and display_all and optim_method == 'sgd' and (n_evals in [100, 300] or n_evals % display_it_nb == 0) :
        #    print(" umin = {}, umax = {}".format(u.min().item(), u.max().item()) )
        #    slices(u, rescale = True)
        #    if flow_type != 'cons' :
        #        slices(-uu.grad, rescale = True)
        if display_all and optim_method == 'bfgs' :
            if n_evals >= next_disp :
                next_disp += display_it_nb
                print("at BFGS iteration {} and eval {},   umin = {}, umax = {}".format(iteration, n_evals, u.min().item(), u.max().item()) )
                slices(u, rescale = True)
                slices(-uu.grad, rescale = True)

    if not nan_OK :
        print('\n \n TERMINATED ON NANs.')

    if check_viable and not viable_OK :
        print('\n \nNOT VIABLE PARAMETERS.')
        print('The shape computed does not satisfy one of the following conditions:')
        print('u.max > .1, u.min < -.1, ratio_discr < .75 \n')
        print('u.min = {:.3f} u.max = {:.3f} ratio = {:.3f}'.format(u.min().item(), u.max().item(), rat))

    t2 = time.time()

    '''Save and show the results'''
    if nan_OK :
        E_curve = np.array(E_curve)
        slices(u.detach().cpu().numpy())

        fig, ax = plt.subplots(2,2, figsize = (12,12))

        ax[0,0].plot(fill_curve_nb * np.arange(len(E_curve)),E_curve)
        if np.array(E_curve).min() > 0 :
            ax[0,0].set_yscale("log")
        ax[0,0].set_title('E curve')

        ax[0,1].plot(fill_curve_nb * np.arange(len(grad_max_curve)), grad_max_curve)
        ax[0,1].set_yscale("log")
        ax[0,1].set_title('grad_max_curve')

        ax[1,0].plot(fill_curve_nb * np.arange(len(grad_L1mean_curve)), grad_L1mean_curve)
        ax[1,0].set_yscale("log")
        ax[1,0].set_title('grad_L1mean_curve')

        ax[1,1].plot(fill_curve_nb * np.arange(len(M_curve)),M_curve)
        ax[1,1].set_title('mass_curve')

        #fig.savefig(snapshot_folder + exp_title + title + ' curves.png')
        #plt.show()

        if False :
            fig, ax = plt.subplots(1,2, figsize = (12,6))

            ax[0].plot(fill_curve_nb * np.arange(len(E_curve)),E_curve)
            if np.array(E_curve).min() > 0 :
                ax[0].set_yscale("log")
            ax[0].set_title('E curve')

            ax[1].plot(fill_curve_nb * np.arange(len(E_curve)),E_curve + 1e-6 - E_curve.min())
            ax[1].set_yscale("log")
            ax[1].set_title('E curve + 1e-6 - minimal value')

            #fig.savefig(snapshot_folder + exp_title + title + ' E_curves.png')
            #plt.show()

        print('')
        print('TOTAL DURATION IN SECONDS', t2 - t1)
        E = loss()
        #breakpoint()
        print('E = {:.4f}'.format(E.item()))

        print(title)
        print('dx = {:.3f} LZ = {:.2f} LX = {:.3f} LY = {:.3f} xi = {}'.format(delta_x,LZ,LX,LY,xi))
        print(optim_props)

        # Open a file with access mode 'a'
        #file_object = open(snapshot_folder + 'meta.txt', 'w')
        # for string in ['Duration = {} seconds'.format(t2 - t1),
        #                'E = {:.4f}'.format(E),
        #                title,
        #                'delta_x = {:.3f} LZ = {:.2f} LX = {:.2f} LY = {:.2f} xi = {}'.format(delta_x, LZ,LX,LY, xi),
        #                str(optim_props), '\n'] :
        #     file_object.write(string + '\n')
        # file_object.close()
    del GradHessConv_ZXY
    torch.cuda.empty_cache()

    if return_var :
        return u, variable

    if check_viable :
        #breakpoint()
        return u, viable_OK, E

    if return_energy :
        #breakpoint()
        return u, E.item()
    #breakpoint()
    return u
