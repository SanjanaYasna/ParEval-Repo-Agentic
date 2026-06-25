/* ============================================================================
 * Barnes-Hut main
 *
 * Usage:
 *   ./barnes-hut --nparticles <NUM> --nsteps <NUM> [--hpx:threads=<NUM>]
 * ============================================================================ */
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "octree.hpp"

constexpr double WIDTH = 99999.0;
constexpr double LENGTH = 99999.0;
constexpr double HEIGHT = 99999.0;

constexpr double ORIGIN_X = -99999.0;
constexpr double ORIGIN_Y = -99999.0;
constexpr double ORIGIN_Z = -99999.0;

namespace
{
int parse_int(std::string const& s)
{
    try {
        std::size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) {
            return -1;
        }
        return v;
    } catch (...) {
        return -1;
    }
}

int parse_threads(std::string const& s)
{
    if (s == "all" || s == "ALL") {
        unsigned hw = std::thread::hardware_concurrency();
        return hw == 0 ? 1 : static_cast<int>(hw);
    }
    int v = parse_int(s);
    return v <= 0 ? 1 : v;
}

int sanitize_threads(int nthreads)
{
    unsigned hw = std::thread::hardware_concurrency();
    int max_threads = (hw == 0) ? 64 : static_cast<int>(hw);
    if (max_threads < 1) {
        max_threads = 1;
    }

    if (nthreads <= 0) {
        return max_threads;
    }
    return std::max(1, std::min(nthreads, max_threads));
}

template <typename F>
void parallel_for(int begin, int end, int nthreads, F&& f)
{
    int n = end - begin;
    if (n <= 0) {
        return;
    }

    if (nthreads <= 1 || n < 2048) {
        for (int i = begin; i < end; ++i) {
            f(i);
        }
        return;
    }

    int workers = std::min(nthreads, n);
    std::atomic<int> next(begin);

    auto worker = [&]() {
        for (;;) {
            int i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= end) {
                break;
            }
            f(i);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers - 1));

    for (int t = 0; t < workers - 1; ++t) {
        try {
            threads.emplace_back(worker);
        } catch (...) {
            break;
        }
    }

    worker();

    for (auto& th : threads) {
        th.join();
    }
}
} // namespace

int main(int argc, char* argv[])
{
    int npart = 0;
    int t_step = 0;

    unsigned hw = std::thread::hardware_concurrency();
    int nthreads = hw == 0 ? 1 : static_cast<int>(hw);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--nparticles" && i + 1 < argc) {
            npart = parse_int(argv[++i]);
        } else if (arg.rfind("--nparticles=", 0) == 0) {
            npart = parse_int(arg.substr(13));
        } else if (arg == "--nsteps" && i + 1 < argc) {
            t_step = parse_int(argv[++i]);
        } else if (arg.rfind("--nsteps=", 0) == 0) {
            t_step = parse_int(arg.substr(9));
        } else if (arg.rfind("--hpx:threads=", 0) == 0) {
            nthreads = parse_threads(arg.substr(14));
        } else if (arg == "--hpx:threads" && i + 1 < argc) {
            nthreads = parse_threads(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            nthreads = parse_threads(argv[++i]);
        } else if (arg.rfind("--threads=", 0) == 0) {
            nthreads = parse_threads(arg.substr(10));
        }
    }

    nthreads = sanitize_threads(nthreads);

    if (npart <= 0 || t_step <= 0) {
        std::cerr << "Usage: barnes-hut --nparticles <NUM> --nsteps <NUM> [--hpx:threads=N]\n";
        return 1;
    }

    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
    }

    std::vector<Particle> particle_array(static_cast<std::size_t>(npart));

    srand(42);
    generate_random_particles(particle_array.data(), space, npart);

    for (int step = 1; step <= t_step; ++step) {
        Octree* octree = create_empty_octree(space);
        for (int j = 0; j < npart; ++j) {
            if (in_space(space, particle_array[j])) {
                octree_insert(octree, space, &particle_array[j]);
            }
        }

        parallel_for(0, npart, nthreads, [&](int idx) {
            particle_array[idx].force_x = 0.0;
            particle_array[idx].force_y = 0.0;
            particle_array[idx].force_z = 0.0;
        });

        parallel_for(0, npart, nthreads, [&](int idx) {
            Particle* tmp_p = &particle_array[idx];
            if (in_space(space, particle_array[idx])) {
                compute_force(tmp_p, octree);
            }
        });

        parallel_for(0, npart, nthreads, [&](int idx) {
            Particle* tmp_p = &particle_array[idx];
            if (in_space(space, particle_array[idx])) {
                update_particle_position_and_velocity(tmp_p);
            }
        });

        free_octree(octree);

        print_summary(particle_array.data(), npart, space, step, 1);
    }

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart << " | Num Steps: " << t_step << "\n";
    }

    return 0;
}
