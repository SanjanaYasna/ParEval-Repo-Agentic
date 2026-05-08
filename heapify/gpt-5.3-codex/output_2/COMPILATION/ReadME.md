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

                