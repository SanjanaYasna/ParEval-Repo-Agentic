# ASTM (Asynchronous Software Transaction Memory)

A header-only Software Transactional Memory (STM) library built on Legion. ASTM provides optimistic concurrency control for shared variables and features such as:

Transaction manager:
    - Provides transaction-local storage
    - Implement the four basic operations which occur in transactions: read, write, async, commit 

## Programs
binary_tree
Example implementation of an unbalanced binary tree implemented using ASTM, with values that have side-effecting constructors. Each non-empty node has astm::new to allocate its value so there are no dependencies, and abstracts programmers away from performing explicit reads and writes through the use of smart pointer syntax. Results are written to binary_tree.txt.

unit_tests
Unit tests covering core STM operations: read/write ordering, overwrites, arithmetic with local variables, future-based reads, and transaction retry logic on conflict. Results are written to unit_tests.txt.

concurrency_tests
Concurrency stress tests verifying atomicity under contention, including threads incrementing a shared integer and squaring transactions. Results are written to concurrency_tests.txt. 

## Prerequisites 

- Legion installed
- A C++17-compliant C++ compiler (e.g. GCC, Clang) and beyond
- Boost, which path is set as `BOOST_ROOT` and incorporated into `LD_LIBRARY_PATH`

### Build
-----------------
Repository depends on Legion 25.06.0

Make sure to set $LEGION_ROOT to Legion install directory 

```
make
```

If you would like to rebulid, run  ```make clean``` before running ```make``` again. 

### Run
The program produces three executables:
    - ```binary_tree``` from binary_tree.cpp , outputs  binary_tree.txt
    - ```unit_tests``` from unit_tests.cpp, outputs unit_tests.txt
    - ```concurrency_tests``` from concurrency_tests.cpp, outputs concurrency_tests.txt
