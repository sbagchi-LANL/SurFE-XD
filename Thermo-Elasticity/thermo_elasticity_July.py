from fenics import *
import matplotlib.pyplot as plt
import numpy as np
import sys

def create_new_mesh_top(k_mesh, lam, A, topsurf, xmax, xmin, ymax, ymin, surftype='gaussian'):

    """create new mesh by mapping a given bottomsurface to the refined kernel mesh
    Args:
        k_mesh: Kernel mesh
        lam: wavelength
        A: amplitude
        topsurf: 1D numpy array containing surface profile
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

    print(len(x_vec))

    if surftype == 'gaussian' and topsurf is None:
        b = lam/np.pi
        topsurf = np.exp(-((x_vec-x1/2)/b)**2)*2*A + ymax
    elif surftype == 'sinusoidal' and topsurf is None:
        w = np.pi/lam
        topsurf = A * np.cos(2 * w * x_vec) + ymax

    #print(x_vec)
    if len(x_vec) != len(topsurf):
        print(topsurf)
    
    yb = np.interp(xnew, x_vec, topsurf, period=(x1 - x0))

    x_bot = xnew[y==1.]
    botsurf = ymin*np.ones((len(x_bot),))
    #print(x_top)
    ys = np.interp(xnew, x_bot, botsurf, period=(x1 - x0))

    ynew = yb + y*(ys - yb)

    xy_new_coor = np.array([xnew, ynew]).transpose()
    k_mesh.coordinates()[:] = xy_new_coor

    return k_mesh


def thermo_elasticity(mesh, L, H, time, num_step, lam, amp, j):
    
    #Boundary conditions
    class PeriodicBoundary(SubDomain):

        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

        # Map right boundary (H) to left boundary (G)
        def map(self, x, y):
            y[0] = x[0] - L
            y[1] = x[1]
    
    class top(SubDomain): # This will later on become a free surface
        def __init__(self):
            SubDomain.__init__(self)

        def inside(self, x, on_boundary):
            return (x[1] > (H - DOLFIN_EPS)) and on_boundary

    # class top(SubDomain):
    #     """ """
    #     def inside(self, x, on_boundary):
    #         """

    #         :param x: 
    #         :param on_boundary: 

    #         """
    #         return x[1] > H - 1e-12 and on_boundary
    
    def lateral_sides(x, on_boundary):
        return (near(x[0], 0) or near(x[0], L)) and on_boundary
    def bottom(x, on_boundary):
        return near(x[1], 0) and on_boundary

    #strain/stress tensor
    def eps(v):
        return sym(grad(v))
    def sigma(v, dT):
        return (lmbda*tr(eps(v))- alpha*(3*lmbda+2*mu)*dT)*Identity(2) + 2.0*mu*eps(v)

    #mark boundary
    boundaries = MeshFunction("size_t", mesh, 1)
    #sub_domains.set_all(6)
    Top = top()
    Top.mark(boundaries, 1)
    x = mesh.coordinates().reshape((-1, 2))
    vs_bot = list(set(sum((f.entities(0).tolist() for f in SubsetIterator(boundaries, 1)), [])))
    #ds_bot=x[vs_bot]

    #Thermoelasticity constant
    a = 1
    sig = 3
    kappa = 386*10e-9
    rho_cp = 385*8940*(10e-9)**3 #specific heat capacity * mass density
    E = Constant(120e9)

    nu = Constant(0.34)
    mu = E/2/(1+nu)
    lmbda = E*nu/(1+nu)/(1-2*nu)
    alpha = Constant(16.7e-6)

    #setting up thermal source
    dt = time/num_step
    center = str(L/2)
    height = str(H)
    rho =  '1.68e-8'
    lam_s = str(lam/np.pi)
    amp_s = str(amp)
    s_profile = '(exp(-pow((x[0]-' + center + ')/' + lam_s + ',2))*' + amp_s + ' + ' + height + ')'
    expr = rho + ' * pow(current*exp((x[1]-' + s_profile + ')/skin),2.)'
    f = Expression(expr, degree=1, a=0.05e-6, current=j, skin=1e2)

    #veriational formulation for time-dependent heat equation
    const_temp = Constant(25.0)
    pbc = PeriodicBoundary()
    VT = FunctionSpace(mesh, "CG", 1, constrained_domain = pbc)
    rhoJ2 = interpolate(f, VT)
    T_D = Expression('0*x[0] + 0*x[1]', degree=2)
    T_n = interpolate(T_D, VT) 
    T_, dT = TestFunction(VT), TrialFunction(VT)
    Delta_T = Function(VT, name="Temperature increase")
    aT = rho_cp*dT*T_*dx + kappa*dt*dot(grad(dT), grad(T_))*dx
    LT = (rho_cp*T_n + dt*f)*T_*dx
    bcT = DirichletBC(VT, Constant(0.), bottom)

    #variational formulation for elasticity
    fu = Constant((0,0))
    Vu = VectorFunctionSpace(mesh, 'CG', 1, constrained_domain = pbc)
    v_tensor = TensorFunctionSpace(mesh, "CG", 1)
    du = TrialFunction(Vu)
    u_ = TestFunction(Vu)
    Wint = inner(sigma(du, Delta_T), eps(u_))*dx
    aM = lhs(Wint)
    LM = rhs(Wint) + inner(fu, u_)*dx
    bcu = [DirichletBC(Vu, Constant((0., 0.)), bottom)]
    u = Function(Vu, name="Displacement")

    #time loop
    loop=0
    t = 0
    for n in range(num_step):
        if n==(num_step-1):
            plt.figure(loop+100, figsize=(9,5)); plt.xlabel('x [m]') ; plt.ylabel('z [m]')
            p = plot(Delta_T,title="$\Delta T$: t = " + str(round(t/1e-9))+"ns")
            plt.colorbar(p)
            plt.show()
        # plt.savefig("plots/thermo_t_" + str(round(t/1e-9))+"ns.png")
        loop+=1

        # Update current time
        t += dt
        #t_list.append(t)

        # Compute solution
        solve(aT == LT, Delta_T, bcT)
        T_n.assign(Delta_T)
    
        #solve(aM == LM, u, bcu)
    
        #s_e = 1/2*inner(sigma(u, Delta_T), eps(u))
        #strain = project(s_e, VT)
        #s_b = np.array([strain(xi) for xi in x[vs_bot]])
        # if n==(num_step):
        #     plt.figure(loop+200, figsize=(9,5))
        #     plt.plot(x[vs_bot][:,0],s_b)
        #     plt.show()

        print(loop)

    Delta_T = interpolate(const_temp, VT)
    # plt.figure(figsize=(9,5)); plt.xlabel('x [m]') ; plt.ylabel('z [m]')
    # p = plot(Delta_T,title="$\Delta T$: t = " + str(round(t/1e-9))+"ns")
    # plt.colorbar(p)
    # plt.show()
    return Delta_T #s_b, x[vs_bot], vs_bot, Delta_T

def elasticity(mesh, L, H, t_dist, J, lam, amp, vs_bot=None, plotting=True, symm=True):

    #Boundary conditions
    class PeriodicBoundary(SubDomain):

        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

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

    #strain/stress tensor
    def eps(v):
        return sym(grad(v))
    def sigma(v, dT):
        return (lmbda*tr(eps(v))- alpha*(3*lmbda+2*mu)*dT)*Identity(2) + 2.0*mu*eps(v)

    #mark boundary
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
    if vs_bot is None:
        vs_bot = list(set(sum((f.entities(0).tolist() for f in SubsetIterator(boundaries, 1)), [])))

    #print(len(vs_bot))

    E = Constant(130e9)
    nu = Constant(0.34)
    mu = E/2/(1+nu)
    lmbda = E*nu/(1+nu)/(1-2*nu)
    alpha = Constant(16e-6)

    pbc = PeriodicBoundary()
    VT = FunctionSpace(mesh, "CG", 1, constrained_domain = pbc)
    Delta_T = Function(VT, name="Temperature increase")
    if len(Delta_T.vector()[:]) == len(t_dist):
        Delta_T.vector()[:] = t_dist
    else:
        print("The dimension does not match up")
        #exit()

    fu = Constant((0,0))
    Vu = VectorFunctionSpace(mesh, 'CG', 2, constrained_domain = pbc)

    traction = Function(Vu)
    traction.vector()[:] = np.zeros((np.size(traction.vector().get_local())))

    v_tensor = TensorFunctionSpace(mesh, "CG", 1)
    du = TrialFunction(Vu)
    u_ = TestFunction(Vu)
    Wint = inner(sigma(du, Delta_T), eps(u_))*dx
    aM = lhs(Wint)
    LM = rhs(Wint) + inner(fu, u_)*dx
    bcu = DirichletBC(Vu,Constant((0.0,0.0)), boundaries,4)
    u = Function(Vu, name="Displacement")

    solve(aM == LM, u, bcu)#, solver_parameters={'linear_solver': 'gmres','preconditioner': 'ilu'})#, 'monitor_convergence':True})
    
    s_e = 1/2*inner(sigma(u, Delta_T), eps(u))
    strain_energy = project(s_e, VT)
    ds_bot=x[vs_bot]
    s_b = np.array([strain_energy(xi) for xi in x[vs_bot]])
    u_b = np.array([u(xi) for xi in x[vs_bot]])
    s_b = s_b[np.argsort(ds_bot[:,0])]
    u_b = u_b[np.argsort(ds_bot[:,0])]*np.sqrt(lam/100e-9)
    ds_bot = ds_bot[np.argsort(ds_bot[:,0])]
    
    
    J_bulk = Function(VT)
    J_bulk.vector()[:] = np.ones(len(J_bulk.vector()[:])) * J

    J_top = Function(VT)
    J_bulk.vector()[vs_bot] = 0.

    if symm is True:
        #print(np.size(s_b))
        N_sb = (np.size(s_b))
        if N_sb % 2 == 0:
            s_b[int(N_sb/2):] = np.flip(s_b[:int(N_sb/2)])
        else:
            s_b[(int(N_sb/2)+1):] = np.flip(s_b[:int(N_sb/2)])


    if plotting == True:
        plt.figure(figsize=(9,5))
        p = plot(sigma(u, Delta_T)[1,1], title='vertical stress')
        plt.colorbar(p)
        plt.show()

        plt.figure(figsize=(9,5))
        p = plot(sigma(u, Delta_T)[0,0], title='horizontal stress')
        plt.colorbar(p)
        plt.show()

        plt.figure(figsize=(9,5))
        p = plot(u[1], title='vertical displacement')
        plt.colorbar(p)
        plt.show()

        plt.figure(figsize=(9,5))
        p = plot(strain_energy, title='strain energy density')
        plt.colorbar(p)
        plt.show()

    return s_b, u_b, ds_bot, vs_bot, J_bulk

def elasticity_build(mesh, L, H, t_dist, lam, amp, vs_bot=None, plotting=True, symm=False):

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

    #strain/stress tensor
    def eps(v):
        return sym(grad(v))
    def sigma(v, dT):
        return (lmbda*tr(eps(v))- alpha*(3*lmbda+2*mu)*dT)*Identity(2) + 2.0*mu*eps(v)

    #mark boundary
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
    if vs_bot is None:
        vs_bot = list(set(sum((f.entities(0).tolist() for f in SubsetIterator(boundaries, 1)), [])))

    #print(len(vs_bot))

    E = Constant(130e9)
    nu = Constant(0.34)
    mu = E/2/(1+nu)
    lmbda = E*nu/(1+nu)/(1-2*nu)
    alpha = Constant(16e-6)

    pbc = PeriodicBoundary()
    VT = FunctionSpace(mesh, "CG", 1, constrained_domain = pbc)
    Delta_T = Function(VT, name="Temperature increase")
    if len(Delta_T.vector()[:]) == len(t_dist):
        Delta_T.vector()[:] = t_dist
    else:
        print("The dimension does not match up")
        #exit()

    fu = Constant((0,0))
    Vu = VectorFunctionSpace(mesh, 'CG', 1, constrained_domain = pbc)

    traction = Function(Vu)
    traction.vector()[:] = np.zeros((np.size(traction.vector().get_local())))

    v_tensor = TensorFunctionSpace(mesh, "CG", 1)
    du = TrialFunction(Vu)
    u_ = TestFunction(Vu)
    Wint = inner(sigma(du, Delta_T), eps(u_))*dx
    aM = lhs(Wint)
    LM = rhs(Wint) + inner(fu, u_)*dx
    bcu = DirichletBC(Vu,Constant((0.0,0.0)), boundaries,4)

    #Set linear Solver parameter
    # PETScOptions.set("ksp_pc_side", "right")
    prec = PETScPreconditioner('hypre_amg')
    # PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
    solver = PETScKrylovSolver('cg', prec)
    solver.parameters['absolute_tolerance'] = 1e-12
    solver.parameters['relative_tolerance'] = 1e-12
    solver.parameters['maximum_iterations'] = 100
    solver.parameters['monitor_convergence'] = True

    u = Function(Vu, name="Displacement")
    A = assemble(aM)
    b = assemble(LM)
    A_petsc = as_backend_type(A)
    b_petsc = as_backend_type(b)
    u_petsc = as_backend_type(u.vector())
    bcu.apply(A_petsc, b_petsc)
    solver.set_operator(A_petsc)
    solver.solve(u_petsc, b_petsc)

    bcu.apply(A_petsc, b_petsc)
    solver.set_operator(A_petsc)
    solver.solve(u_petsc, b_petsc)
    #solve(aM == LM, u, bcu, solver_parameters={'linear_solver': 'gmres','preconditioner': 'ilu'})#, 'monitor_convergence':True})
    
    s_e = 1/2*inner(sigma(u, Delta_T), eps(u))
    strain_energy = project(s_e, VT, solver_type='gmres', preconditioner_type='hypre_amg')
    ds_bot=x[vs_bot]
    u_b = np.array([u(xi) for xi in x[vs_bot]])
    s_b = np.array([strain_energy(xi) for xi in x[vs_bot]])
    u_b = u_b[np.argsort(ds_bot[:,0])]#*np.sqrt(lam/100e-9)
    s_b = s_b[np.argsort(ds_bot[:,0])]
    ds_bot = ds_bot[np.argsort(ds_bot[:,0])]

    if symm is True:
        #print(np.size(s_b))
        N_sb = (np.size(s_b))
        if N_sb % 2 == 0:
            s_b[int(N_sb/2):] = np.flip(s_b[:int(N_sb/2)])
        else:
            s_b[(int(N_sb/2)+1):] = np.flip(s_b[:int(N_sb/2)])

    if plotting == True:
        plt.figure(figsize=(9,5))
        p = plot(sigma(u, Delta_T)[1,1], title='vertical stress')
        plt.colorbar(p)
        plt.show()

        plt.figure(figsize=(9,5))
        p = plot(sigma(u, Delta_T)[0,0], title='horizontal stress')
        plt.colorbar(p)
        plt.show()

        plt.figure(figsize=(9,5))
        p = plot(u[1], title='vertical displacement')
        plt.colorbar(p)
        plt.show()

        plt.figure(figsize=(9,5))
        p = plot(strain_energy, title='strain energy density')
        plt.colorbar(p)
        plt.show()

    return s_b, u_b, ds_bot, vs_bot

def get_uniform_temp(mesh, L, H, const_temp):
    class PeriodicBoundary(SubDomain):

        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

        # Map right boundary (H) to left boundary (G)
        def map(self, x, y):
            y[0] = x[0] - L
            y[1] = x[1]
    
    pbc = PeriodicBoundary()
    VT = FunctionSpace(mesh, "CG", 1, constrained_domain = pbc)
    DT = Function(VT)
    DT = interpolate(Constant(const_temp), VT)
    D_T = DT.vector()[:]

    return D_T

def plane_stress(mesh, L, H, t_dist, lam, amp, T_app):

    #Boundary conditions
    class PeriodicBoundary(SubDomain):

        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

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
            return x[1] > H - 1e-12 and on_boundary
    
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

    def bottom(x, on_boundary):
        return near(x[1], 0) and on_boundary

    #strain/stress tensor
    def eps(v):
        return sym(grad(v))
    def sigma(v, dT):
        return (lmbda*tr(eps(v))- alpha*(3*lmbda+2*mu)*dT)*Identity(2) + 2.0*mu*eps(v)

    #mark boundary
    boundaries = MeshFunction("size_t", mesh, 1)
    #sub_domains.set_all(6)
    Top = top()
    Top.mark(boundaries, 1)
    Left = left()
    Left.mark(boundaries, 2)
    Right = right()
    Right.mark(boundaries, 3)
    ds=Measure('ds',domain=mesh,subdomain_data=boundaries)
    x = mesh.coordinates().reshape((-1, 2))
    vs_bot = list(set(sum((f.entities(0).tolist() for f in SubsetIterator(boundaries, 1)), [])))

    #print(len(vs_bot))

    #Thermoelasticity constant
    a = 1
    sig = 3
    kappa = 386
    rho_cp = 385*8940 #density * specific heat capacity

    E = Constant(120e3)
    nu = Constant(0.34)
    mu = E/2/(1+nu)
    lmbda = E*nu/(1+nu)/(1-2*nu)
    alpha = Constant(16.7e-6)

    pbc = PeriodicBoundary()
    VT = FunctionSpace(mesh, "CG", 1)#, constrained_domain = pbc)
    Delta_T = Function(VT, name="Temperature increase")
    if len(Delta_T.vector()[:]) == len(t_dist):
        Delta_T.vector()[:] = t_dist
    else:
        print("The dimension does not match up")
        #exit()

    T_D = Expression('0*x[0] + 0*x[1]', degree=2)
    Delta_T = interpolate(T_D, VT)
    fu = Constant((0,0))
    Vu = VectorFunctionSpace(mesh, 'CG', 2)#, constrained_domain = pbc)

    plt.figure(figsize=(9,5))
    p = plot(Delta_T)
    plt.colorbar(p)
    plt.show()

    traction = Function(Vu)
    traction.vector()[:] = np.zeros((np.size(traction.vector().get_local())))
    #traction = Constant((0,E*alpha*T_app))

    traction_left = Constant((E*T_app*alpha, 0.))
    traction_right = Constant((-E*T_app*alpha, 0.))
    #print(120e3*16.7e-6*T_app)

    v_tensor = TensorFunctionSpace(mesh, "CG", 1)
    du = TrialFunction(Vu)
    u_ = TestFunction(Vu)
    Wint = inner(sigma(du, Delta_T), eps(u_))*dx
    aM = lhs(Wint)
    LM = rhs(Wint) + inner(fu, u_)*dx + inner(traction,(u_))*ds(1) + inner(traction_left, u_)*ds(2) + inner(traction_right, u_)*ds(3)
    #LM = rhs(Wint) + inner(fu, u_)*dx
    bcu = [DirichletBC(Vu.sub(1), Constant(0.), bottom)]
    u = Function(Vu, name="Displacement")

    solve(aM == LM, u, bcu)
    
    s_e = 1/2*inner(sigma(u, Delta_T), eps(u))
    strain_energy = project(s_e, VT)
    ds_bot=x[vs_bot]
    s_b = np.array([strain_energy(xi) for xi in x[vs_bot]])
    #s_b = s_b[np.argsort(ds_bot[:,0])]
    #ds_bot = ds_bot[np.argsort(ds_bot[:,0])]

    strain = project(s_e, VT)
    s_b = np.array([strain(xi) for xi in x[vs_bot]])
    plt.figure(figsize=(9,5))
    p = plot(sigma(u, Delta_T)[1,1], title='vertical stress')
    plt.colorbar(p)
    plt.show()

    plt.figure(figsize=(9,5))
    p = plot(sigma(u, Delta_T)[0,0], title='horizontal stress')
    plt.colorbar(p)
    plt.show()

    plt.figure(figsize=(9,5))
    p = plot(u[1], title='vertical displacement')
    plt.colorbar(p)
    plt.show()

    plt.figure(figsize=(9,5))
    p = plot(strain_energy, title='strain energy density')
    plt.colorbar(p)
    plt.show()

    return s_b, ds_bot, vs_bot
