#%%
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from thermo_elasticity_July import *
from electrodiffuse_2 import *
import cProfile, pstats

#%matplotlib ipympl

#%matplotlib ipympl
def main():
    lam = 100
    as_ratio = float(sys.argv[3])
    A = as_ratio*lam/2
    dh_ratio = 0.5
    
    xlen = 5*max(2*A, lam)
    xmax = xlen
    xmin = xmax - xlen
    ylen = dh_ratio*xlen
    ymax = ylen#2*max(2*A, lam)
    ymin = 0.

    Nx = 220+1
    Ny = int((Nx-1)*dh_ratio)+1 
    
    refinement_level = 2   # for aspect ratio smaller than 0.5, this could be set to 1, otherwise set this 2. (also for more high precision solution on refined meshes turn 'fine=True' in electrostatics_2 (i.e. solve with 'CG2' elements)--Note that this will considerably slow down the electric field solver!
    #############
    gamma = 1.
    gamma_e = 1. # to compute critical e-field
    perm = 1.
    d_om = 1.
    ymodulus = 130e9
    alpha = 16e-6
    ##############
    e_cr = np.sqrt(2*np.pi*gamma_e/lam/perm) # critical E-field for growth
    e_applied = e_cr*float(sys.argv[2])
    ############################################################
    kernel_mesh = piecewise_mesh(Nx, Ny, dh_ratio, refinement_level)
    # kernel_mesh = UnitSquareMesh(601,50,'crossed')
    # s_profile = np.loadtxt('random_surface_small.txt')
    # mesh_s = create_new_mesh_top(Mesh(kernel_mesh), lam, A, s_profile[:,1]+ylen, xlen, 0., ylen, 0.)
    # mesh_e = create_new_mesh_bot(Mesh(kernel_mesh), lam, A, s_profile[:,1], xlen, 0., ylen, 0.)
    mesh_s = create_new_mesh_top(Mesh(kernel_mesh), lam, A, None, xlen, 0., ylen, 0., surftype='gaussian')
    mesh_e = create_new_mesh_bot(Mesh(kernel_mesh), lam, A, None, xlen, 0., ylen, 0., surftype='gaussian')
    # plt.figure(figsize=(9,5))
    # plot(mesh_s)
    # plt.savefig('mesh_test.png')
    # plt.figure(figsize=(9,5))
    # plot(mesh_e)
    # plt.show()
    #t_applied = np.sqrt(np.pi/lam)*float(sys.argv[1])
    t_applied = np.sqrt(np.pi*gamma_e*ymodulus/lam)/(ymodulus*alpha)*float(sys.argv[1])
    #print(t_applied)
    DT = get_uniform_temp(mesh_s, xlen, ylen, t_applied)

    #s_surf, disp, dsurf, index = elasticity(mesh_s, xlen, ylen, DT, lam, 2*A, plotting=False, symm=True)
    s_surf, disp, dsurf, index = elasticity_build(mesh_s, xlen, ylen, DT, lam, 2*A, plotting=False, symm=False)
    #N_sb = (np.size(s_surf))
    #print(s_surf[150], s_surf[151])
    #print(s_surf[int(N_sb/2):] - np.flip(s_surf[:int(N_sb/2)]))

    Esurf, dsurf, surf_ind = electrostatics_2(mesh_e, e_applied, xmax, xmin, ymax, ymin, None, symm=False)
    print(Esurf[0], Esurf[-1])
    # plt.figure(figsize=(9,5)), plt.title('strain density')
    # plt.plot(dsurf[:,0],s_surf)
    # plt.show()
    # plt.figure(figsize=(9,5)), plt.title('E^2')
    # plt.plot(dsurf[:,0],Esurf)
    # plt.show()
    # plt.figure(figsize=(9,5))
    # plt.plot(dsurf[:,0],disp[:,0]), plt.title('Horizontal Displacement ($10^{-7}$)')
    # plt.show()

    n = len(dsurf)
    #print(n)
    t = 0
    dt = 1e-2 ###### change
    max_dy_allowed = 1e-5 # depends on geometry (change accordingly)
    min_dy_allowed = 2e-6
    # Tf=200*dt
    ic = 0
    o_freq = 1000 ###### change
    e_solve_freq = 100 ###### change
    run_steps = 1000000 #5000000 # 1000000
    #############################################################
    while ic<= run_steps:
        xn, yn, vn, k = fdm_surface_evol_nonuniform(dsurf, disp, Esurf, s_surf, d_om, gamma, perm, lam, dt)
        # xn, yn, vn, k = fem_surface_evol(dsurf, Esurf, s_surf, d_om, gamma, perm, lam, dt) # Surface_evol(dsurf, Esurf, d_om, gamma, perm, lam, dt)
        # del_y = max(abs(vn[int(n/2-n/3):int(n/2+n/3)]))*dt
        del_y = max(vn)*dt
        # plt.figure(figsize = (9,5)), plt.title('FinDiff Velocity')
        # plt.plot(xn, vn, '-')
        # plt.savefig('velocity_test.png')
        # plt.show()
        #print(del_y)
        ## Stability Check ##
        if del_y>max_dy_allowed:
            dt/=2
            xn, yn, vn, k = fdm_surface_evol_nonuniform(dsurf, disp, Esurf, s_surf, d_om, gamma, perm, lam, dt)
            print(max(vn), "dt/100")
        elif del_y<min_dy_allowed:
            dt*=1.5
            xn, yn, vn, k = fdm_surface_evol_nonuniform(dsurf, disp, Esurf, s_surf, d_om, gamma, perm, lam, dt)
        dsurf = np.column_stack((xn,yn)) # updated surface
        t+=dt
        ic+=1
        mass = total_area(xn,yn)
        if np.mod(ic,o_freq)==0:
            config_output(xn,yn,vn,t)
            log_output_variable(t,mass)
            print(ic)
            # plt.figure()
            # plt.plot(xn, vn)
            # plt.show()
            # plt.savefig('velocity_'+'{:.6f}'.format(t)+'.png')

        if np.mod(ic, e_solve_freq)==0:
            mesh_s = create_new_mesh_top(Mesh(kernel_mesh), lam, A, dsurf[:,1]+ylen, xmax, xmin, ylen, 0.)
            mesh_e = create_new_mesh_bot(Mesh(kernel_mesh), lam, A, dsurf[:,1], xmax, xmin, ymax, ymin)
            #s_surf, vdisp, dsurf, index = elasticity(mesh_s, xlen, ylen, DT, lam, 2*A, vs_bot=index, plotting=False)
            s_surf, disp, dsurf, index = elasticity_build(mesh_s, xlen, ylen, DT, lam, 2*A, vs_bot=index, plotting=False, symm=False)
            Esurf, dsurf, surf_ind = electrostatics_2(mesh_e, e_applied, xmax, xmin, ymax, ymin, None, vs_bot=surf_ind, symm=False)
            # plt.figure(figsize=(9,5)), plt.title('surface velocity')
            # plot(mesh_s)
            # plt.show()
            # print(dsurf[:,1])
    print("all done")
    # print(dsurf[:,1]+500.)

if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.print_stats() 
    # stats.dump_stats('profile_data')

# %%
