#!/home/sbagchi/miniconda3/envs/fenicsproject python
# coding: utf-8

from electrodiffuse import *
import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
        
    lam = 200
    as_ratio = float(sys.argv[1])
    A = as_ratio*lam/2
    dh_ratio = float(sys.argv[3])
 
    xlen = 10*max(2*A, lam)
    xmax = xlen
    xmin = xmax - xlen
    ylen = dh_ratio*xlen
    ymax = 2*max(2*A, lam)
    ymin = ymax - ylen

    Nx = 100+1
    Ny = int((Nx-1)*dh_ratio)+1 
  
    refinement_level = 1   # for aspect ratio smaller than 0.5, this could be set to 1, otherwise set this 2. (also for more high precision solution on refined meshes turn 'fine=True' in electrostatics_2 (i.e. solve with 'CG2' elements)--Note that this will considerably slow down the electric field solver!
    #############
    gamma = 1.
    gamma_e = 1. # to compute critical e-field
    perm = 1.
    d_om = 1.
    ##############
    e_cr = np.sqrt(2*np.pi*gamma_e/lam/perm) # critical E-field for growth
    e_applied = float(sys.argv[2])*e_cr
    ############################################################
    kernel_mesh = piecewise_mesh(Nx, Ny, dh_ratio, refinement_level)
    mesh = create_new_mesh(Mesh(kernel_mesh), lam, A, None, xmax, xmin, ymax, ymin)
#    surf_ind = np.loadtxt('bottom_boundary_index_'+repr(Nx)+'_'+repr(Ny)+'_'+repr(refinement_level)+'.txt').astype(int)
    Esurf, dsurf, surf_ind = electrostatics_2(mesh, e_applied, xmax, xmin, ymax, ymin, None)
    #############################################################
    t = 0
    dt = 1e-3 ###### change
    max_dy_allowed = 1e-5 # depends on geometry (change accordingly)
    # Tf=200*dt
    ic = 0
    o_freq = 1000 ###### change
    e_solve_freq = 100 ###### change
    run_steps = 5000000 # 1000000
    #############################################################
    while ic<= run_steps:
        xn, yn, vn, k = fdm_surface_evol(dsurf, Esurf, d_om, gamma, perm, lam, dt) # Surface_evol(dsurf, Esurf, d_om, gamma, perm, lam, dt)
        _,_, v_surf, _ = fdm_surface_evol(dsurf, Esurf, d_om, gamma, 0., lam, dt)
        _, _, v_elec, _ = fdm_surface_evol(dsurf, Esurf, d_om, 0., perm, lam, dt)
        del_y = max(abs(vn))*dt
        ## Stability Check ##
        if del_y>max_dy_allowed:
            dt/=100
            xn, yn, vn, k = fdm_surface_evol(dsurf, Esurf, d_om, gamma, perm, lam, dt)
            _,_, v_surf, _ = fdm_surface_evol(dsurf, Esurf, d_om, gamma, 0., lam, dt)
            _, _, v_elec, _ = fdm_surface_evol(dsurf, Esurf, d_om, 0., perm, lam, dt)
        dsurf = np.column_stack((xn,yn)) # updated surface
        t+=dt
        ic+=1
        mass = total_area(xn,yn)
        if np.mod(ic,o_freq)==0:
            config_output(xn,yn,vn,t)
            np.savetxt('vel_surf.{:.6f}'.format(t), v_surf, delimiter=' ')
            np.savetxt('vel_elec.{:.6f}'.format(t), v_elec, delimiter=' ')
            log_output_variable(t,mass)
#            plt.figure()
#            plt.plot(xn, vn)
#            plt.savefig('velocity_'+'{:.6f}'.format(t)+'.png')

        if np.mod(ic, e_solve_freq)==0:
            mesh = create_new_mesh(Mesh(kernel_mesh), lam, A, dsurf[:,1], xmax, xmin, ymax, ymin)
            Esurf, dsurf, surf_ind = electrostatics_2(mesh, e_applied, xmax, xmin, ymax, ymin, lam, vs_bot=surf_ind)
        print (np.size(dsurf[:,0]))
           
print ("All Done !!!")
