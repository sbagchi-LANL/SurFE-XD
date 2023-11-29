# © 2023. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are.

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare.

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit.

# others to do so.

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from dolfin import *
#from fenics import *
import numpy as np
import scipy as scp
from scipy import integrate
import matplotlib.pyplot as plt
from pandas import *
from findiff import *
import time
import scipy.signal as sg



def par_findiff(xn, yn):
    """findiff operation for piecewise-uniform data
    Args:
        xn: x coordinates of piecewise surface
        yn: y coordinates of piecewise surface
    
    Returns:
        k : local curvature
        s : local ds--surface differential for peicewise data
    """
    
    d_dx = FinDiff(0, xn[2]-xn[1], acc=4)
    dy_dx = d_dx(yn)
    d2_x = FinDiff(0, xn[2]-xn[1], 2, acc=4)
    d2y_dx = d2_x(yn)
    k = d2y_dx/(1+dy_dx**2)**1.5
    s = 1./np.sqrt(1+dy_dx**2)
    return k, s

def curvature_normal_refined(x, y):
    
    """Computes global curvature and ds by indentifying interval
    for piecewise (x,y) data by using par_findiff.
    Args:
        x : full x coordinates for surface
        y : full y coordinates for surface
    Returns:
        k : Gathered curvature for total surface with concave +ve and convex -ve
        s : Gathered ds for toal surface
        indx : indicies associated with piecewise intervals

    """
    
    x_eff= x[x.argsort()]
    y_eff= y[x.argsort()]
    
    del_x = x_eff[1:] - x_eff[:-1]
    del2_x = del_x[1:] - del_x[:-1]
    
    indx = np.append(np.append(0, sg.find_peaks(abs(del2_x), threshold=.01)[0]+2), np.size(x_eff))
    
    k = np.zeros((np.size(x_eff)))
    s = np.zeros((np.size(x_eff)))
    
    for i in range(np.size(indx)-1):
            k[indx[i]:indx[i+1]], s[indx[i]:indx[i+1]] = par_findiff(x_eff[indx[i]:indx[i+1]], y_eff[indx[i]:indx[i+1]])
    
    return k, s, indx


def subdomain_refine(mesh, i):
    """Refines mesh at a subdomain (hard-coeded) for given reniement level i

    Args:
        mesh : dolfin mesh-object
        i: refinement level
    Returns:
        refined mesh object
    Raises:
        Valueerror if the level of refinement exceeds 2
    """
    
    x = mesh.coordinates()[:,0]
    y = mesh.coordinates()[:,1]
    x_bot = x[y==0]
    x_bot=x_bot[x_bot.argsort()]
    N = int(np.size(x_bot)/2)
    y_mid=y[x==x_bot[N]]
#     print ("N:", N)
    
    if i==1 :
        xlo = x_bot[N-int((N-1)*0.4)-1]
        xhi = x_bot[int((N-1)*0.4)+N]
        yhi = y_mid[int(np.size(y_mid)/2)]
    elif i==2:
        xlo = x_bot[N-int(N*0.3)]
        xhi = x_bot[N+int(N*0.3)]
        yhi = y_mid[int(np.size(y_mid)/2)]
    else:
        raise ValueError("expected mesh refinement level less than 3")

    
    class fine(SubDomain):
        """ """
        def inside(self, x, on_boundary):
            """

            :param x: 
            :param on_boundary: 

            """
            return x[0] >= xlo and x[0] <= xhi and x[1] <= yhi #2. - y_bound

    sub_domains_bool = MeshFunction("bool", mesh, mesh.topology().dim())
    sub_domains_bool.set_all(False)
    fine().mark(sub_domains_bool, True)
    return refine(mesh, sub_domains_bool)


def piecewise_mesh(hl, vl, dh_ratio, refine_num):
    """Creates a kernel piecewise mesh-object which can be used to map orbitrary geometries
    Arg:
        hl: num. of horizontal layers
        vl: num. of vertical layers
        dh_ratio: domain height/breadth ratio
        refine_num: levels of mesh refinement
    Returnes:
        kernel mesh-object
    """
    
    breadth = 10.0
    height = dh_ratio*breadth
    
    mesh = RectangleMesh(Point(0.0, 0.0), Point(breadth, height), hl, vl,'crossed')
    for i in range(refine_num):
#         print ('Nx_new :', np.size(mesh.coordinates()[:,0]))
        mesh = subdomain_refine(mesh, i+1)

    # Extract and normalize x and y coordinates from mesh
    x = mesh.coordinates()[:, 0]/breadth
    y = mesh.coordinates()[:, 1]/height
    
    xy_new_coor = np.array([x, y]).transpose()
    mesh.coordinates()[:] = xy_new_coor
    return mesh


def create_new_mesh(k_mesh, lam, A, bottomsurf, xmax, xmin, ymax, ymin):
    
    """create new mesh by mapping a given bottomsurface to the refined kernel mesh
    Args:
        k_mesh: Kernel mesh
        lam: wavelength
        A: amplitude
        bottomsurf: 1D numpy array containing surface profile
        xmax:
        xmin:
        ymax:
        ymin:
    Returns:
        mapped mesh-object

    """
    
    x0 = xmin #min(x)
    x1 = xmax #max(x)
    
    x = k_mesh.coordinates()[:,0]
    y = k_mesh.coordinates()[:,1]
    
    # Map coordinates on unit square to the computational domain
    xnew = x0 + x*(x1 - x0)
    x_vec = xnew[y==0.]
    x_vec = x_vec[np.argsort(x_vec)]
    
    if bottomsurf is None:
        b = lam/np.pi
        bottomsurf = 2*np.exp(-((x_vec-x1/2)/b)**2)*A + ymin
    
    yb = np.interp(xnew, x_vec, bottomsurf, period=(x1 - x0))
    
    x_top = xnew[y==1.]
    topsurf = ymax*np.ones((len(x_top),))
    ys = np.interp(xnew, x_top, topsurf, period=(x1 - x0))
    
    ynew = yb + y*(ys - yb)
    
    xy_new_coor = np.array([xnew, ynew]).transpose()
    k_mesh.coordinates()[:] = xy_new_coor
    
    return k_mesh



def electrostatics_2(mesh, e, xmax, xmin, ymax, ymin, lam, vs_bot=None, symm=True, fine=False):
    """Electrostatics solver for a given E-field as Neumann BC at top boundary and 0 potential as Dirichilet BC
    at bottom boundary of the vacuum domain

    Args:
        mesh: vacuum domain mesh-object
        e: applied E-field
        xmax:
        xmin:
        ymax:
        ymin:
        lam: wavelength
        vs_bot: indicies associated with bottom surface (Default value = None)
        symm:  Whether the geometry is symmetric or not (Default value = True)
        fine:  To compute finer E-field solution with 'CG2" elements
        (otherwise 'CG1" is used for faster solutions
        (Default value = False)
    Returns:
        Es_bot = E^2 at the bottom surface
        ds_bot = coordinates of the bottom surface
        vs_bot= indices associated with bottom surface
    """
    tol = 1e-12

    class bottom(SubDomain):
        """ """
        def inside(self, x, on_boundary):
            """

            :param x: 
            :param on_boundary: 

            """
            return on_boundary

    # Sub domain for inflow (right)
    class right(SubDomain):
        """ """
        def inside(self, x, on_boundary):
            """

            :param x: 
            :param on_boundary: 

            """
            return x[0] > xmax - tol and on_boundary

    # Sub domain for outflow (left)
    class left(SubDomain):
        """ """
        def inside(self, x, on_boundary):
            """

            :param x: 
            :param on_boundary: 

            """
            return x[0] < xmin + tol and on_boundary

    # Sub domain for outflow (top)
    class top(SubDomain):
        """ """
        def inside(self, x, on_boundary):
            """

            :param x: 
            :param on_boundary: 

            """
            return x[1] > ymax - tol and on_boundary
    
    
    
    class PeriodicBoundary(SubDomain):
        """ """

    # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            """

            :param x: 
            :param on_boundary: 

            """
            return bool(x[0] < tol and x[0] > -tol and on_boundary)

    # Map right boundary (H) to left boundary (G)
        def map(self, x, y):
            """

            :param x: 
            :param y: 

            """
            y[0] = x[0] - (xmax - xmin)
            y[1] = x[1]

    
    
    pbc = PeriodicBoundary(tol)
    interior = MeshFunction("size_t", mesh, 2)
    boundaries = MeshFunction("size_t", mesh, 1)
    #sub_domains.set_all(6)
    Bot = bottom()
    Bot.mark(boundaries, 1)
    Top = top()
    Top.mark(boundaries, 2)
    Left = left()
    Left.mark(boundaries, 3)
    Right = right()
    Right.mark(boundaries, 4)

    # #####################################
    
    # Define finite elements spaces
    if fine is True:
        V=FunctionSpace(mesh,'CG', 2, constrained_domain=pbc)
    else:
        V=FunctionSpace(mesh,'CG', 1, constrained_domain=pbc)

    #####################################

    u_b_1=DirichletBC(V,Constant(0.0), boundaries,1)
#     u_b_2=DirichletBC(V,Constant(1.0),boundaries,2)
    bc = u_b_1 #, u_b_2]
    
    dx=Measure('dx',domain=mesh,subdomain_data=interior)
    ds=Measure('ds',domain=mesh,subdomain_data=boundaries) # For Neumann domain

    ########################################################
    u=TrialFunction(V)
    v=TestFunction(V)
    a=dot(grad(u),grad(v))*dx
    g_y=-e
    g=Constant((g_y))
    L=Constant(0.0)*v*dx + g*v*ds(2) + Constant(0.0)*v*ds(3)  + Constant(0.0)*v*ds(4)
    #####################################

    # Set linear solver parameters
    PETScOptions.set("ksp_pc_side", "right")
    prec = PETScPreconditioner('hypre_amg')
    PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
    solver = PETScKrylovSolver('cg', prec)
    solver.parameters['absolute_tolerance'] = 1e-16
    solver.parameters['relative_tolerance'] = 1e-16
    solver.parameters['maximum_iterations'] = 100
    solver.parameters['monitor_convergence'] = True

    # Compute solution
    u = Function(V)
#     solve(a == L, u, bc, solver_parameters=prm)
    A = assemble(a)
    b = assemble(L)
    A_petsc = as_backend_type(A)
    b_petsc = as_backend_type(b)
    u_petsc = as_backend_type(u.vector())
    bc.apply(A_petsc, b_petsc)
    solver.set_operator(A_petsc)
    solver.solve(u_petsc, b_petsc)
#    solve(a==L,u,bc) #, solver_parameters={'linear_solver': 'gmres','preconditioner': 'ilu'})
    V_e = FunctionSpace(mesh,'CG', 1) #, constrained_domain=pbc)
    E = -grad(u)

# E-field norm-square at various boundary regions
    E2 = project(inner(E,E), V_e, solver_type='gmres', preconditioner_type='hypre_amg')
#     E_mag = project(sqrt(inner(E,E)), V_e)
#    p=plot(E2) #E_mag)
#    p.set_cmap("rainbow")
#    plt.colorbar(p)
#    plt.savefig('E_field_'+repr(lam)+'.png', bbox_inches = 'tight')

    x = mesh.coordinates().reshape((-1, 2))
    if vs_bot is None:
        vs_bot = list(set(sum((f.entities(0).tolist() for f in SubsetIterator(boundaries, 1)), [])))
    ds_bot=x[vs_bot]
    
    E_b = np.array([E2(xi) for xi in x[vs_bot]])
    # sorted Es
    E_b = E_b[np.argsort(ds_bot[:,0])]
    ds_bot = ds_bot[np.argsort(ds_bot[:,0])]
    if symm is True:
        E_b[int((np.size(E_b)-1)/2+1):] = np.flip(E_b[:int((np.size(E_b)+1)/2-1)])
    return  E_b, ds_bot, vs_bot #, Es ,ds_1

# In[6]:


def Imesh(x_dof):
    
    """create 1-D interfacial mesh from bottom surface dofs

    :param x_dof: 

    """
    
    s_c = []
    for i in range(len(x_dof)-1):
        s_c.append([i,i+1])
    s_cells = np.asarray(list(np.asanyarray(s_c)))
    
    x_mesh = Mesh()
    editor = MeshEditor()
    editor.open(x_mesh, 'interval', 1, 1)
    editor.init_vertices(len(x_dof))
    editor.init_cells(len(s_cells))

    [editor.add_vertex(i, [n]) for i,n in enumerate(x_dof)]
    [editor.add_cell(i, n) for i,n in enumerate(s_cells)]

    editor.close()
    return x_mesh


def fem_surface_evol(ds, e2, d_om, gamma, perm, lam, dt):
    """Surface evolver routine using FEM
    Args:
        ds: coordinates of surface profile (2D numpy array)
        e2: E^2 corresponding to the surface
        d_om: Diffusivity*(Atomic-Volume)**2*(number_fof_atoms) in unit cross section of flow
        gamma: surface tension
        perm: electrical permittivity of vacuum
        lam: wavelength
        dt: timestep
    Returns:
        xnew: updated x-coordinates of surface (in this framework this is useless)
        ynew: updated y-coordinates of surface
        vn: velocity along y-direction at each mesh point in surface (1D numpy array)
        k: curvature
    """
    
    
    x_dof = ds[:,0]
    y_dof = ds[:,1]
    k, s = curvature_normal_refined(x_dof, y_dof)
    # sorting based on x-dofs for evaluating derivatives
    e2 = e2[x_dof.argsort()]
    x_dof = x_dof[x_dof.argsort()]
    y_dof = y_dof[x_dof.argsort()]
    # free energy density function
    f = d_om*(-gamma*k - 0.5*perm*e2) # 
#    # surface gradient
#    d_dx = FinDiff(0, lam/Nh)
#    dy_dx = d_dx(y_dof)
#    s = 1/(1.+dy_dx**2)**0.5 # ds/dx
    # implementing surface evolution with FEM
    tol = 1e-8
    # Sub domain for outflow (left)
    xmin = min(x_dof)
    xmax = max(x_dof)
    # Sub domain for inflow (right)
    class right(SubDomain):
        """ """
        def inside(self, x, on_boundary):
            """

            :param x: 
            :param on_boundary: 

            """
            return x[0] > xmax - tol and on_boundary

    # Sub domain for outflow (left)
    class left(SubDomain):
        """ """
        def inside(self, x, on_boundary):
            """

            :param x: 
            :param on_boundary: 

            """
            return x[0] < xmin + tol and on_boundary


    class PeriodicBoundary(SubDomain):
        """ """

    # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            """

            :param x: 
            :param on_boundary: 

            """
            return bool(x[0] < tol and x[0] > -tol and on_boundary)

    # Map right boundary (H) to left boundary (G)
        def map(self, x, y):
            """

            :param x: 
            :param y: 

            """
            y[0] = x[0] - (xmax - xmin)
   #        y[1] = x[1]
  
    pbc = PeriodicBoundary(tol)
    x_mesh = Imesh(x_dof)
    boundaries = MeshFunction("size_t", x_mesh, 0)
    Left = left()
    Left.mark(boundaries, 3)
    Right = right()
    Right.mark(boundaries, 4)
    
    P1 = FiniteElement("Lagrange", x_mesh.ufl_cell(), 1)
#    S1 = FiniteElement("DG", x_mesh.ufl_cell(), 0)
    mixed = MixedElement([ P1, P1])
    V = FunctionSpace(x_mesh, mixed, constrained_domain=pbc)
    
    F = Function(V.sub(0).collapse())
    F.vector()[:] = f
    
    S = Function(V.sub(0).collapse())
    S.vector()[:] = s
    
#    j_bcL=DirichletBC(V.sub(0),Constant(0.0),boundaries,3)
#    j_bcR=DirichletBC(V.sub(0),Constant(0.0),boundaries,4)
    bc = [] # [j_bcL, j_bcR]
     # Trial functions
    j,vn = TrialFunctions(V)
    # Test functions
    phi,w = TestFunctions(V)
    
    # Bilnear form
    a = j*phi/S*dx + vn*w*dx - j*w.dx(0)*dx  
    # Linear Form
    L = phi.dx(0)*F*dx 
    # # define function to hold solution
    h_ = Function(V)
    
    # Set linear solver parameters
    prm = LinearVariationalSolver.default_parameters()
    linear_solver='Krylov'
    if linear_solver == 'Krylov':
        prm["linear_solver"] = 'gmres'
        prm["preconditioner"] = 'ilu'
        prm["krylov_solver"]["absolute_tolerance"] = 1e-16
        prm["krylov_solver"]["relative_tolerance"] = 1e-16
        prm["krylov_solver"]["maximum_iterations"] = 10000 #max_iter
    else:
        prm["linear_solver"] = 'lu'
    
#    # Define linear variational problem
#    pde = LinearVariationalProblem(a, L, h_, bc)
#    solver = LinearVariationalSolver(pde)
#    # Solve variational problem
#    solver.solve()

    solve(a==L, h_, bc, solver_parameters=prm)
    j_, vn_ = split(h_)
    
    j = project(j_,V.sub(0).collapse())
    vn = project(vn_,V.sub(0).collapse()).vector().get_local()
    
    xnew = x_dof # + vn.vector().get_local()*n[:,0]*dt
    ynew = y_dof + vn*dt # *n[:,1] 
    return xnew, ynew, vn, k #j.vector().get_local()


def fdm_surface_evol(ds, e2, d_om, gamma, perm, lam, dt):
    """Surface evolver routine using finite difference
    Args:
        ds: coordinates of surface profile (2D numpy array)
        e2: E^2 corresponding to the surface
        d_om: Diffusivity*(Atomic-Volume)**2*(number_fof_atoms) in unit cross section of flow
        gamma: surface tension
        perm: electrical permittivity of vacuum
        lam: wavelength
        dt: timestep
    Returns:
        xnew: updated x-coordinates of surface (in this framework this is useless)
        ynew: updated y-coordinates of surface
        vn: velocity along y-direction at each mesh point in surface (1D numpy array)
        k: curvature
    """
    
    x_dof = ds[:,0]
    y_dof = ds[:,1]
    e2 = e2[x_dof.argsort()]
    y_dof = y_dof[x_dof.argsort()]
    x_dof = x_dof[x_dof.argsort()]

    
    k, s, indx = curvature_normal_refined(x_dof, y_dof)
    
    F = d_om*(-gamma*k - 0.5*perm*e2)
    vfn = np.zeros(np.size(x_dof))
    
    for i in range(np.size(indx)-1):
        d_dx = FinDiff(0, x_dof[indx[i]+1]-x_dof[indx[i]])
        df_dx = d_dx(F[indx[i]:indx[i+1]])
        df_ds = df_dx*s[indx[i]:indx[i+1]]
        vfn[indx[i]:indx[i+1]] = d_dx(df_ds)
    
    yn = y_dof + vfn*dt
    xn = x_dof
    return xn, yn, vfn, k


def total_area(x,y):
    """Computes total area under a given curve y(x)

    :param x: 
    :param y: 

    """
    Ar = integrate.simps(y+100,x) # reference height at -100. 
    return Ar
    

def config_output(x,y,v,t):
    """Dumps x, y, and velocties of surface points at time t

    :param x: 
    :param y: 
    :param v: 
    :param t: 

    """
    X = np.column_stack((x,y,v))
    np.savetxt('surf.'+'{:.6f}'.format(t)+'.txt',X, delimiter=' ')


def log_output_variable(t,m):
    """Creates log output of any timeseries variable

    :param t: 
    :param m: 

    """
    with open('log_area.txt','a') as file:
        file.write(repr(t)+' '+repr(m)+'\n')
    return
        
