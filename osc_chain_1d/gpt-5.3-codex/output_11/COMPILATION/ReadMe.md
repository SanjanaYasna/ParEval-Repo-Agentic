# Legion Oscillator Chain (ODEINT)

The program integrates a 1D nonlinear oscillator chain with:

- On-site potential exponent: `KAPPA = 3.5`
- Coupling exponent: `LAMBDA = 4.5`

Time integration is performed using boost::numeric::odeint::symplectic_rkn_sb3a_mclachlan which is a stepper suitable for Hamiltonian systems. (https://www.boost.org/doc/libs/1_88_0/boost/numeric/odeint/stepper/symplectic_rkn_sb3a_mclachlan.hpp)
All energy calculation outputs are written to ```odeint.txt```. The compiled program executable should be called ```odeint```.

### Prerequisites

- Legion installed
- A C++17-compliant C++ compiler (e.g. GCC, Clang) and beyond
- Boost, which path is set as `BOOST_ROOT` and incorporated into `LD_LIBRARY_PATH`

### Build

Repository depends on Legion 25.06.0

Make sure to set $LEGION_ROOT to Legion install directory 

```
mkdir build; cd build

cmake ..

make
```

### Run
-----------------
The program is controlled by four command-line parameters:

--N (int)
Total dimension of the oscillator chain. Default 1024

--G (int)
Block size (number of elements per task, officially termed as 'dataflow' as this was a translation from HPX). Defult 128

--steps (int)
Number of time integration steps. Default 100

--dt (double) 
Time step size, default 0.01

The energy computations should return a rounded integer results of the computation. 

Example Run:
```bash
> ./odeint --N 2048 --dt 0.1
> cat odeint.txt
Dimension: 2048, number of elements per dataflow: 128, number of dataflow: 16, steps: 100, dt: 0.1
Initialization complete, energy: 341
Integration complete, energy: 341
```
