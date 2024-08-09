# controlling active polar fluid in bulk


import os
import random
from fenics import *
from dolfin import *
#from mshr import *
import numpy as np
import sys

comm = MPI.comm_world
rank = MPI.rank(comm)

learning_rate = 0.1
A, B, C, D, E = 1e-10, 0.1, 0.1, 1.2, 1.2

set_log_level(50)


nx, ny = 60, 60
LX, LY = 128, 128
count = 0
mesh = RectangleMesh(Point(0, 0), Point(LX, LY), nx, ny)



dt = 1
time = 1800
num_steps = int(time/dt)
dal = 50
epsilon = 0.04
penalty = []
noise = 3
learning = []

# model parameters

w_0 = 0.05
lmd = 0.8

w_0_star = 0.2236

# data_dir = f"/scratch1/saptorshighosh/control/dec14/D_{D}"
# folder = f"aster_LX_{LX}_nx_{nx}_{time}_t_lr_{learning_rate}"
# parent_dir = f'{data_dir}/{folder}'

data_dir = f"/home/fenics/shared"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
nopy = False

# if not os.path.exists(parent_dir):
#     os.makedirs(parent_dir)
# nopy = False

####################################################################################################


class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 0) and near(x[1], LY)) or
                          (near(x[0], LX) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], LX) and near(x[1], LY):
            y[0] = x[0] - LX
            y[1] = x[1] - LY
        elif near(x[0], LX):
            y[0] = x[0] - LX
            y[1] = x[1]
        else:   # near(x[1], 127)
            y[0] = x[0]
            y[1] = x[1] - LY


#####################################################################################################
V = VectorElement("Lagrange", mesh.ufl_cell(), 1, dim=2)
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
MFS = FunctionSpace(mesh, MixedElement([Q, V]), constrained_domain=PeriodicBoundary())
FS = FunctionSpace(mesh, Q, constrained_domain=PeriodicBoundary())
#####################################################################################################


u_star = Function(MFS)
(rho_star, tau_star) = split(u_star)

data = HDF5File(comm, f"aster_40_70.h5", 'r')
data.read(u_star, f"u_new/Vector/vector_1")

w = Function(FS)


class InitialConditions_w(UserExpression):
    def __init__(self, **kwargs):
        random.seed(1 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = np.sqrt(w_0)


w_init = InitialConditions_w(degree=1)
w.interpolate(w_init)


w_all_k = []


for i in range(num_steps):

    w_all_k.append(w.copy(deepcopy = True))


w_star = Function(FS)

class InitialConditions_wstar(UserExpression):
    def __init__(self, **kwargs):
        random.seed(1 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = w_0_star

w_init_ = InitialConditions_wstar(degree = 1)
w_star.interpolate(w_init_)
#########################################################


# state functions

u_new = Function(MFS)
u_old = Function(MFS)
(rho_new, tau_new) = split(u_new)
(rho_old, tau_old) = split(u_old)


# adjoint functions

v_new = Function(MFS)
v_old = Function(MFS)

(eta_new, nu_new) = split(v_new)
(eta_old, nu_old) = split(v_old)


# test functions
tf = TestFunction(MFS)
(q, v) = split(tf)



# functional derivative of lagrangian with respect to omega

#djdw = C*(w - w_star) - 2*w*dot(tau_new, grad(eta_new)) - 2*w*rho_new*div(nu_new)

djdw = C*(2*w)*(w**2 - w_star**2) - 2*w*dot(tau_new, grad(eta_new)) - 2*w*rho_new*div(nu_new)


def timestep(w_all_k):


    if(rank == 0):
        print(f"count:{count}", flush = True)

    u_all = []


    for i in range(num_steps):
        
        if(rank == 0 and i%10 == 0):
            print(f"forward:{i}", flush = True)

        a2 = (1 - rho_new)
        a4 = (1 + rho_new)/pow(rho_new, 2)


        w.assign(w_all_k[i])

        Res_0 = (rho_new - rho_old)/dt*q*dx + \
                dot(grad(rho_new), grad(q))*dx + div(w**2*tau_new)*q*dx

        Res_1 = dot((tau_new - tau_old), v)/dt*dx + (a2 + a4*dot(tau_new, tau_new))*dot(tau_new, v)*dx + \
                dot(grad(w**2*rho_new), v)*dx + inner(nabla_grad(tau_new), nabla_grad(v))*dx - \
                lmd*(0.5*dot(grad(dot(tau_new, tau_new)), v)*dx + div(tau_new)*dot(tau_new, v)*dx -
                dot(dot(tau_new, nabla_grad(tau_new)), v)*dx)

        Res = Res_0 + Res_1
        u_trial = TrialFunction(MFS)

        J = derivative(Res, u_new, u_trial)

        problem = NonlinearVariationalProblem(Res, u_new, [], J)
        solver = NonlinearVariationalSolver(problem)
        #prm = solver.parameters

        solver.solve()

        u_all.append(u_old.copy(deepcopy = True))
        u_old.assign(u_new)


    v_all = []

    u_new.assign(u_all[-1])

    a1 = -1
    a3 = -(2 + rho_new)/rho_new**3


    v_old = project( -A*(u_new - u_star), MFS)

    for i in range(num_steps):

        j = num_steps - i - 1

        if(rank == 0 and i%10 == 0):
            print(f"backward:{j}", flush = True)

        u_new.assign(u_all[j])
        w.assign(w_all_k[j])

        Res3 = ((eta_old - eta_new)*q/dt*dx - D*(rho_new - rho_star)*q*dx - dot(grad(eta_new), grad(q))
                * dx + w**2*div(nu_new)*q*dx - (a1 + a3*dot(tau_new, tau_new))*(dot(nu_new, tau_new))*q*dx)

        Res4 = (dot((nu_old - nu_new), v)/dt*dx - E*dot((tau_new - tau_star), v)*dx - inner(nabla_grad(nu_new), nabla_grad(v))*dx + w**2*dot(grad(eta_new), v)*dx
                - (a2 + a4*dot(tau_new, tau_new))*dot(nu_new, v)*dx - 2*a4*dot(nu_new, tau_new)*dot(tau_new, v)*dx +
                lmd*(-div(nu_new)*dot(tau_new, v)*dx + 2*div(tau_new)*dot(nu_new, v)*dx - dot(grad(dot(nu_new, tau_new)), v)*dx + dot(dot(tau_new, grad(nu_new)), v)*dx
                     - dot(dot(grad(tau_new), nu_new), v)*dx))

        Res_adj = Res3 + Res4
        solve(Res_adj == 0, v_new)

        # v_all[:, j] = v_old.vector()[:]
        v_all.append(v_old.copy(deepcopy = True))

        v_old.assign(v_new)

    v_all.reverse()

    return u_all, v_all


def update_control(u_all, v_all, w_all_k, lr):

    wt = Function(FS)
    updated_w = []

    for i in range(num_steps):

        if(rank == 0 and i%10 == 0):
            print(f"Update:{i}", flush = True)

        u_new.assign(u_all[i])
        v_new.assign(v_all[i])
        w.assign(w_all_k[i])

        wt.assign( project( (w - lr*djdw), FS) )
        updated_w.append(wt.copy(deepcopy = True))

    return updated_w

############################################################################################

# COST CALCULATION


def cost_function(u_all, w_all_k, w_star):
    """
    cost_function(u_all)

    Function to compute the cost for a given trajectory
    """


    # Terminal state penalty


    u_new.assign(u_all[-1])

    cost_functional = (0.5 * A * (rho_new - rho_star)**2
                       + 0.5 * B * dot((tau_new - tau_star),
                                       (tau_new - tau_star))**2
                       + 0.5 * C * dot(grad(rho_new), grad(rho_new))) * dx

    terminal_cost = assemble(cost_functional)
    # stage state penalty

    stage_state_cost = 0.0
    #print("Start stage calculation")
    for i in range(num_steps):

        u_new.assign(u_all[i])
        w.assign(w_all_k[i])

        cost_functional = (0.5 * D * (rho_new - rho_star)**2
                           + 0.5 * E * dot((tau_new-tau_star),
                                           (tau_new-tau_star))**2
                           + 0.5 * C * (w**2 - w_star**2)**2) * dx

        stage_state_cost += dt * assemble(cost_functional)

    return (terminal_cost + stage_state_cost)


def armijo(u_all, v_all, w_all_k, w_star):

    cost = 0.0
    
    gradf = C*(w - w_star) - 2*w*dot(tau_new, grad(eta_new)) - 2*w*rho_new*div(nu_new)

    for i in range(num_steps):
        

        u_new.assign(u_all[i])
        v_new.assign(v_all[i])
        w.assign(w_all_k[i])

        Res = -dot(gradf, gradf)
        #test = project(Res, FS)
        cost_functional = Res*dx
        cost += dt*assemble(cost_functional)

    return cost



while(count < dal):

    Hdf = HDF5File(mesh.mpi_comm(), f'{data_dir}/data_test_count_{count}_w_{w_0_star}_time_{time}.h5', "w")
    Hdf.write(mesh, "mesh")

    if(rank == 0):
        print(f"count:{count}")
    
    lr = learning_rate
   
    data.read(u_new, f"u_new/Vector/vector_0")
    data.read(u_old, f"u_new/Vector/vector_0") 

    (u_all, v_all) = timestep(w_all_k)
    penalty.append(cost_function(u_all, w_all_k, w_star))
    
    # Backtracking implementation
    if(count != 0):
        
        while((penalty[count] > penalty[count - 1]) and lr > 0.0001):   

            lr = 0.1*lr
            print(f"Correcting learning rate to {lr}", flush = True)
            w_all_k = update_control(u_backup, v_backup, w_backup, lr)

            data.read(u_new, f"u_new/Vector/vector_0")
            data.read(u_old, f"u_new/Vector/vector_0") 
            (u_all, v_all) = timestep(w_all_k)
            penalty[count] = cost_function(u_all, w_all_k, w_star)
 
        learning.append(lr)   

    # armj = armijo(u_all, v_all, w_all_k, w_star)     
    w_backup = w_all_k.copy()
    (u_backup, v_backup) = (u_all.copy(), v_all.copy())
    w_all_k = update_control(u_all, v_all, w_all_k, lr).copy()  
  

    if(count > 20):
    
        for i in range(num_steps):

            u_new.assign( u_all[i] )
            Hdf.write(u_new, f"u_new/Vector/vector_{i}")
            # v_new.assign( v_all[i] )
            # Hdf.write(v_new, f"v_new/Vector/vector_{i}")
            w.assign( w_all_k[i] )
            Hdf.write(w, f"w/Vector/vector_{i}")
            
    
    Hdf.close()        

    
    if(rank == 0):
        np.savez(f"{data_dir}/cost_w_{w_0_star}_time_{time}.npz", J = penalty, lr = learning)
    count = count + 1


if(rank == 0):
    np.savez(f"{data_dir}/cost_w_{w_0_star}_time_{time}.npz", J = penalty, lr = learning)


data.close()
