/* ============================================================================
 * Barnes-Hut main (standalone C++17)
 * Usage:
 *   ./barnes-hut --nparticles <NUM> --nsteps <NUM> [--hpx:threads=<NUM>]
 * ============================================================================ */
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "octree.hpp"

constexpr double WIDTH   = 99999.0;
constexpr double LENGTH  = 99999.0;
constexpr double HEIGHT  = 99999.0;

constexpr double ORIGIN_X = -99999.0;
constexpr double ORIGIN_Y = -99999.0;
constexpr double ORIGIN_Z = -99999.0;

template <typename F>
void parallel_for(std::size_t begin, std::size_t end, std::size_t num_threads, const F& f)
{
    if (end <= begin) {
        return;
    }

    std::size_t total = end - begin;
    if (num_threads == 0) {
        num_threads = 1;
    }
    num_threads = std::min(num_threads, total);

    if (num_threads <= 1 || total < 4096) {
        for (std::size_t i = begin; i < end; ++i) {
            f(i);
        }
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(num_threads - 1);

    std::size_t base = total / num_threads;
    std::size_t rem = total % num_threads;

    auto run_range = [&](std::size_t s, std::size_t e) {
        for (std::size_t i = s; i < e; ++i) {
            f(i);
        }
    };

    std::size_t start = begin;
    for (std::size_t t = 0; t < num_threads; ++t) {
        std::size_t len = base + (t < rem ? 1 : 0);
        std::size_t stop = start + len;

        if (t + 1 == num_threads) {
            run_range(start, stop);
        } else {
            workers.emplace_back([&, start, stop]() { run_range(start, stop); });
        }

        start = stop;
    }

    for (auto& th : workers) {
        th.join();
    }
}

static bool parse_int_str(const std::string& s, int& out)
{
    try {
        std::size_t pos = 0;
        long long v = std::stoll(s, &pos, 10);
        if (pos != s.size() || v < INT_MIN || v > INT_MAX) {
            return false;
        }
        out = static_cast<int>(v);
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_size_t_str(const std::string& s, std::size_t& out)
{
    try {
        std::size_t pos = 0;
        unsigned long long v = std::stoull(s, &pos, 10);
        if (pos != s.size()) {
            return false;
        }
        out = static_cast<std::size_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

int main(int argc, char* argv[])
{
    int npart = 0;
    int t_step = 0;
    std::size_t nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) {
        nthreads = 1;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--nparticles" && i + 1 < argc) {
            parse_int_str(argv[++i], npart);
        } else if (arg.rfind("--nparticles=", 0) == 0) {
            parse_int_str(arg.substr(std::string("--nparticles=").size()), npart);
        } else if (arg == "--nsteps" && i + 1 < argc) {
            parse_int_str(argv[++i], t_step);
        } else if (arg.rfind("--nsteps=", 0) == 0) {
            parse_int_str(arg.substr(std::string("--nsteps=").size()), t_step);
        } else if (arg == "--hpx:threads" && i + 1 < argc) {
            parse_size_t_str(argv[++i], nthreads);
        } else if (arg.rfind("--hpx:threads=", 0) == 0) {
            parse_size_t_str(arg.substr(std::string("--hpx:threads=").size()), nthreads);
        } else if (arg == "--threads" && i + 1 < argc) {
            parse_size_t_str(argv[++i], nthreads);
        } else if (arg.rfind("--threads=", 0) == 0) {
            parse_size_t_str(arg.substr(std::string("--threads=").size()), nthreads);
        } else {
            // Ignore unknown runtime options for compatibility.
        }
    }

    if (npart <= 0 || t_step <= 0) {
        std::cerr << "Usage: ./barnes-hut --nparticles <NUM> --nsteps <NUM> [--hpx:threads=N]\n";
        return 1;
    }

    if (nthreads == 0) {
        nthreads = 1;
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
            if (in_space(space, particle_array[static_cast<std::size_t>(j)])) {
                octree_insert(octree, space, &particle_array[static_cast<std::size_t>(j)]);
            }
        }

        parallel_for(0, particle_array.size(), nthreads, [&](std::size_t idx) {
            particle_array[idx].force_x = 0.0;
            particle_array[idx].force_y = 0.0;
            particle_array[idx].force_z = 0.0;
        });

        parallel_for(0, particle_array.size(), nthreads, [&](std::size_t idx) {
            if (in_space(space, particle_array[idx])) {
                compute_force(&particle_array[idx], octree);
            }
        });

        parallel_for(0, particle_array.size(), nthreads, [&](std::size_t idx) {
            if (in_space(space, particle_array[idx])) {
                update_particle_position_and_velocity(&particle_array[idx]);
            }
        });

        free_octree(octree);

        print_summary(particle_array.data(), npart, space, step, 1);
    }

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart
                << " | Num Steps: " << t_step << "\n";
    }

    return 0;
}
