# Barnes-Hut Simulation using HPX

More information about the Barnes-Hut Simulation can be found [here](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation).
This implementation contains an HPX version of the simulation that provides net particle simulation statistics at each timestep.

## Programs
main.cpp
Rank 0 computes and writes aggregate statistics to barnes_hut.txt after each timestep, and appends a final summary line with the total number of particles and timesteps. The computations are done with precision of 2. 

octree.hpp
Implements octree data structure and Barnes–Hut force.

particle.hpp
Defines particle and domain representations plus support routines. Prints out global statistics (active count, kinetic energy, total momentum, and center of mass) to be formatted into barnes_hut.txt each timestep.

## Prerequisites

- HPX installed (ideally 1.5.1 or 1.10.0)
- A C++17-compliant C++ compiler (e.g. GCC, Clang) and beyond
- `HPX_LOCATION` set to the HPX installation prefix (the directory containing `lib/`, `include/`, etc.)
- Boost, which path is set as `BOOST_ROOT` and incorporated into `LD_LIBRARY_PATH`

### Build

From the repository root, run

```
make
```

If you would like to rebulid, run  ```make clean``` before running ```make``` again. 

## Run

The program has command-line options:

--nparticles (int)
Number of particles in the simulation. Each particle is initialized with a random position in a fixed 3D domain and equal mass.

--nsteps (int)
Number of simulation timesteps to advance. At each step the octree is rebuilt, forces are computed, and positions/velocities are updated.

```
Example Run: 
```bash
> ./barnes-hut --nparticles 100 --nsteps 5
> cat barnes_hut.txt 
Step    1 | active 100/100 | KE 9.25e+05 | |p| 5.78e+04 | CoM (49.95, 48.81, 54.59)
Step    2 | active 100/100 | KE 3.70e+06 | |p| 1.16e+05 | CoM (49.95, 48.81, 54.59)
Step    3 | active 100/100 | KE 8.33e+06 | |p| 1.73e+05 | CoM (49.95, 48.81, 54.59)
Step    4 | active 100/100 | KE 1.48e+07 | |p| 2.31e+05 | CoM (49.95, 48.81, 54.59)
Step    5 | active 100/100 | KE 2.32e+07 | |p| 2.89e+05 | CoM (49.95, 48.81, 54.59)
Num Particles: 100 | Num Steps: 5
```
```
