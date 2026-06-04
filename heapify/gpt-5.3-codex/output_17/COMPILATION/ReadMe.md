WIP (Work-in-progress) level parallelism based algorithm for the implementation of `make_heap`
within Legion parallel algorithms library. The goal of this prototype is to experiment with a *bottom‑up, level‑parallel* heap
construction strategy using futures and to explore how chunking
and synchronization between tree levels work. All benchmark results are written to `heaps.txt`. The compiled program executable should be called ```heaps```.

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
The heaps benchmark is controlled by two command-line options:

--vector_size (int)
Number of elements in the input vector. The vector is initialized as
0, 1, 2, ..., vector_size - 1.

--chunk_size (int)
Number of heap nodes processed per task at each tree level.
If chunk_size > 0, it directly controls the per-task work size.
If chunk_size == 0 (default), it is chosen automatically as
vector_size / (number of worker threads).

Example Run: 
```bash
> ./heaps --vector_size 999999 --chunk_size 1000
> cat heaps.txt
First 10 elements: 999998 786430 999997 655358 786429 917502 999996 589822 655357 720894
Last 10 elements: 60 124997 3905 1952 975 7811 487 31248 15623 124998
Sum of all elements: 499998500001
Root (max) element: 999998
Is valid heap: true
```
