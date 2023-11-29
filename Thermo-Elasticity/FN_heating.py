import warnings
from dolfin import *
# from fenics import *
import numpy as np
import scipy as scp
from scipy import integrate
import matplotlib.pyplot as plt
from pandas import *
from findiff import *


def calculate_FN_currents(Elocal, phi):
    """.

    Parameters:
    Elocal: (numpy array)
    phi: work function (scalar)

    Returns:
    J_dc : current density at the surface
    """
    # J_expr1_dc contains values preceeding the exponential in dc FN equation for J
    J_expr1_dc = 1.54 * 1e-6 * 10 ** (4.52 * phi ** (-0.5)) * Elocal ** 2 / phi
    # J_expr2_dc contains values in the exponential in dc FN equation for J
    J_expr2_dc = -6.53 * 1e9 * phi ** 1.5 / Elocal
    # Format the two expressions to be FEniCS constants
    J_dc = J_expr1_dc * np.exp(J_expr2_dc)
    # 
    return J_dc


def structure_electrostatics(mesh, J, vs_top=None, type='insulator'):
    tol = 1e-8
    x_dof = mesh.coordinates()[:, 0]
    y_dof = mesh.coordinates()[:, 1]
    xmax = max(x_dof)
    xmin = min(x_dof)
    ymax = max(y_dof)
    ymin = min(y_dof)


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
            return near(x[0], xmax) and on_boundary

    # Sub domain for outflow (left)
    class left(SubDomain):
        """ """

        def inside(self, x, on_boundary):
            """

            :param x:
            :param on_boundary:

            """
            return near(x[0], 0) and on_boundary

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
    ds=Measure('ds',domain=mesh,subdomain_data=boundaries)
    x = mesh.coordinates().reshape((-1, 2))
            
    if vs_top is None:
        vs_top = list(set(sum((f.entities(0).tolist() for f in SubsetIterator(boundaries, 2)), [])))

    if type == 'insulator':
        pbc = PeriodicBoundary(tol)
        V = FunctionSpace(mesh, 'CG', 1)

        J_bulk = Function(V)
        J_bulk.vector()[:] = np.ones(len(x_dof)) * J

        J_top = Function(V)
        J_bulk.vector()[vs_top] = 0.

        return J_bulk, J_top, x[vs_top]

    elif type == 'metal':
        ## Have to implement normal BVP

        # # project current density into function space
        # proj_J = project(J_dc, V) # [A/m^2]
        # # Integrate current density over the vacuum/structure surface boundary
        # dc_current = assemble(proj_J_dc*ds(1)) # [A/m]

        pass


    elif type == 'semiconductor':
        ## polarized field solution inside material needs to be passed
        pass

    
def heating(mesh, L, initial_temp, fermi_energy, rho, C_v, q_e, kappa, J_bulk, J_top, vs_bot=None):

    #Boundary conditions
    class PeriodicBoundary(SubDomain):

        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool(x[0] < 1e-8 and x[0] > -1e-8 and on_boundary)

        # Map right boundary (H) to left boundary (G)
        def map(self, x, y):
            y[0] = x[0] - L
            y[1] = x[1]
    
    class top(SubDomain):
        """ """
        def inside(self, x, on_boundary):
            """

            :param x: 
            :param on_boundary: 

            """
            return on_boundary
    
    class left(SubDomain): # This will later on become a free surface
        def __init__(self):
            SubDomain.__init__(self)

        def inside(self, x, on_boundary):
            return near(x[0], 0) and on_boundary
    
    class right(SubDomain): # This will later on become a free surface
        def __init__(self):
            SubDomain.__init__(self)

        def inside(self, x, on_boundary):
            return near(x[0], L) and on_boundary

    class bottom(SubDomain): # This will later on become a free surface
        def __init__(self):
            SubDomain.__init__(self)

        def inside(self, x, on_boundary):
            return near(x[1], 0) and on_boundary
        
    x_dof = mesh.coordinates()[:, 0]

    #mark boundary
    interior = MeshFunction("size_t", mesh, 2)
    boundaries = MeshFunction("size_t", mesh, 1)
    #sub_domains.set_all(6)
    Top = top()
    Top.mark(boundaries, 1)
    Left = left()
    Left.mark(boundaries, 2)
    Right = right()
    Right.mark(boundaries, 3)
    Bottom = bottom()
    Bottom.mark(boundaries, 4)
    ds=Measure('ds',domain=mesh,subdomain_data=boundaries)
    x = mesh.coordinates().reshape((-1, 2))

    #print(len(vs_bot))

    pbc = PeriodicBoundary()

    V = FunctionSpace(mesh, 'CG', 1, constrained_domain=pbc)
    
    T_d = Constant(initial_temp)
    T_bc = DirichletBC(V, T_d, boundaries, 4)
    dx = Measure('dx', domain=mesh, subdomain_data=interior)
    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)  # For Neumann domain

    # Define initial values
    T_0 = T_d
    T_n = interpolate(T_0, V)  # interpolate into function space

    b = TrialFunction(V)
    q = TestFunction(V)

    # define gradient of T (responsible for Nottingham heating)
    # gradT_n = J_top * Constant(fermi_energy / q_e / kappa)
    gradT_n = J_bulk * Constant(fermi_energy / q_e / kappa)

    # begin stationary heat equation
    
    J_bulk = inner(J_bulk, J_bulk)
    f = J_bulk * Constant(rho)  # rho J^2, Joule heating source term

    # set up BVP
    a = kappa * dot(grad(b), grad(q)) * dx
    L = (f) * q * dx + kappa * gradT_n * q * ds(1)

    T = Function(V)

    solve(a == L, T, T_bc)  # solve
    
    dT = Function(V)
    dT.vector()[:] = T.vector()[:] - initial_temp
    
    return T, dT.vector()[:]
