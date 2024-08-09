
import os
import random
from fenics import *
from dolfin import *
import numpy as np
import sys

comm = MPI.comm_world
rank = MPI.rank(comm)

set_log_level(50)

# generate mesh

nx, ny = 60, 120
LX, LY = 128, 256
mesh = RectangleMesh(Point(0, 0), Point(LX, LY), nx, ny)

dt = 1
time = 10000
num_steps = int(time/dt) + 1

time_save = 2
# model parameters

w_0 = 0.05
lmd = 0.8
seed = 12345

data_dir = f"/home/fenics/shared"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
nopy = False

# data_dir = os.getcwd()


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


w = Function(FS)


class InitialConditions_w(UserExpression):
    def __init__(self, **kwargs):
        random.seed(1 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = w_0


w_init = InitialConditions_w(degree=1)
w.interpolate(w_init)


# state functions

u_new = Function(MFS)
u_old = Function(MFS)
(rho_new, tau_new) = split(u_new)
(rho_old, tau_old) = split(u_old)

rho0 = 1.07

class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(seed + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = rho0 + 0.03*(0.5 - random.random())
        values[1] = 0.00 + 0.03*(0.5 - random.random())
        values[2] = 0.00 + 0.03*(0.5 - random.random())
    def value_shape(self):
        return (3,)

u_init = InitialConditions(degree = 1)
u_new.interpolate(u_init)
u_old.interpolate(u_init)


tf = TestFunction(MFS)
(q, v) = split(tf)

Hdf = HDF5File(comm, f"{data_dir}/forward_w_{w_0}_lmd_{lmd}_time_{time_save}_seed_{seed}.h5", "w")
Hdf.write(mesh, "mesh")

done = False
steps = 0


while not done:


    if(steps > int(time - time_save)):
        Hdf.write(u_old, f"u_new/Vector/vector_{steps}")
        
    if(rank == 0 and steps%10 == 0):
        print(f"forward-step: {steps}", flush = True)

    a2 = (1 - rho_new)
    a4 = (1 + rho_new)/pow(rho_new, 2)


    Res_0 = (rho_new - rho_old)/dt*q*dx + \
            dot(grad(rho_new), grad(q))*dx + div(w*tau_new)*q*dx

    Res_1 = dot((tau_new - tau_old), v)/dt*dx + (a2 + a4*dot(tau_new, tau_new))*dot(tau_new, v)*dx + \
            dot(grad(w*rho_new), v)*dx + inner(nabla_grad(tau_new), nabla_grad(v))*dx - \
            lmd*(0.5*dot(grad(dot(tau_new, tau_new)), v)*dx + div(tau_new)*dot(tau_new, v)*dx -
            dot(dot(tau_new, nabla_grad(tau_new)), v)*dx)

    Res = Res_0 + Res_1

    solve(Res == 0, u_new)
    u_old.assign(u_new)



    done = steps>=num_steps
    steps = steps+1


Hdf.close()
