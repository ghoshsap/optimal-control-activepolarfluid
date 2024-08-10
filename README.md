# Spatiotemporal control of structure and dynamics in a polar active fluid

We apply optimal control theory to a model of a polar active fluid (the Toner-Tu model), with
the objective of driving the system into particular emergent dynamical behaviors or programming
switching between states on demand. We use the effective self-propulsion speed as the control pa-
rameter (i.e. the means of external actuation). We identify control protocols that achieve outcomes
such as relocating asters to targeted positions, forcing propagating solitary waves to reorient to a
particular direction, and switching between stationary asters and propagating fronts. We analyze
the solutions to identify generic principles for controlling polar active fluids. Our findings have
implications for achieving spatiotemporal control of active polar systems in experiments, particu-
larly in vitro cytoskeletal systems. Additionally, this research paves the way for leveraging optimal
control methods to engineer the structure and dynamics of active fluids more broadly.

<img width="510" alt="Screenshot 2024-08-09 at 1 52 58 AM" src="https://github.com/user-attachments/assets/a86cf77b-0366-4446-884c-965cae6b30e7">


### Steps to Run the Script

#### Launching FEniCS Using Docker

To launch FEniCS using Docker, use the following commands:

```bash
docker run -ti -v $(pwd):/home/fenics/shared:z quay.io/fenicsproject/stable
```
#### Running python code

```bash
cd /home/fenics/shared
mpirun -n 4 python3 forward_polar.py
```


### Running the Script on a HPC cluster. Have the docker image in the same folder as the python scripts

#### Need to have singularity using apptainer

```bash
module --ignore-cache load share_modules/APPTAINER/1.3.2
module swap openmpi3 mvapich2/2.2
srun --mpi=pmi2 -n 4 singularity exec -H /home/saptorshighosh/docker_tmp -B data_dir:/home/fenics/shared fenics_stable.img python3 control_polar.py
```




