/* ============================================================================
 * Barnes-Hut main
 * ============================================================================ */
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
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
    bool starts_with(std::string const& s, std::string const& prefix)
    {
        return s.rfind(prefix, 0) == 0;
    }

    bool parse_int(std::string const& s, int& out)
    {
        try
        {
            std::size_t idx = 0;
            int val = std::stoi(s, &idx);
            if (idx != s.size())
            {
                return false;
            }
            out = val;
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    int resolve_num_threads(int requested)
    {
        if (requested > 0)
        {
            return requested;
        }
        unsigned int hw = std::thread::hardware_concurrency();
        if (hw == 0)
        {
            return 1;
        }
        return static_cast<int>(hw);
    }

    template <typename F>
    void parallel_for_indices(int begin, int end, int num_threads, F&& f)
    {
        if (end <= begin)
        {
            return;
        }

        int total = end - begin;
        if (num_threads <= 1 || total <= 1)
        {
            for (int i = begin; i < end; ++i)
            {
                f(i);
            }
            return;
        }

        int chunk = (total + num_threads - 1) / num_threads;
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(num_threads));

        for (int t = 0; t < num_threads; ++t)
        {
            int chunk_begin = begin + t * chunk;
            if (chunk_begin >= end)
            {
                break;
            }
            int chunk_end = std::min(chunk_begin + chunk, end);

            workers.emplace_back([chunk_begin, chunk_end, &f]() {
                for (int i = chunk_begin; i < chunk_end; ++i)
                {
                    f(i);
                }
            });
        }

        for (auto& th : workers)
        {
            th.join();
        }
    }

    void print_usage()
    {
        std::cerr << "Usage: ./barnes-hut [--nparticles <NUM>] [--nsteps <NUM>] "
                     "[--hpx:threads=<NUM>]\n";
    }
}

int main(int argc, char* argv[])
{
    int npart = 100000;
    int t_step = 20;
    int requested_threads = 0;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--nparticles" && i + 1 < argc)
        {
            if (!parse_int(argv[++i], npart))
            {
                print_usage();
                return 1;
            }
        }
        else if (starts_with(arg, "--nparticles="))
        {
            if (!parse_int(arg.substr(std::string("--nparticles=").size()), npart))
            {
                print_usage();
                return 1;
            }
        }
        else if (arg == "--nsteps" && i + 1 < argc)
        {
            if (!parse_int(argv[++i], t_step))
            {
                print_usage();
                return 1;
            }
        }
        else if (starts_with(arg, "--nsteps="))
        {
            if (!parse_int(arg.substr(std::string("--nsteps=").size()), t_step))
            {
                print_usage();
                return 1;
            }
        }
        else if (arg == "--hpx:threads" && i + 1 < argc)
        {
            parse_int(argv[++i], requested_threads);
        }
        else if (starts_with(arg, "--hpx:threads="))
        {
            parse_int(arg.substr(std::string("--hpx:threads=").size()), requested_threads);
        }
        else if (arg == "--nprocs" && i + 1 < argc)
        {
            ++i; // ignored compatibility option
        }
        else if (starts_with(arg, "--nprocs="))
        {
            // ignored compatibility option
        }
    }

    if (npart <= 0 || t_step <= 0)
    {
        print_usage();
        return 1;
    }

    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
    }

    std::unique_ptr<Particle[]> particle_array(new Particle[npart]());
    std::srand(42);
    generate_random_particles(particle_array.get(), space, npart);

    int nthreads = resolve_num_threads(requested_threads);
    nthreads = std::max(1, std::min(nthreads, npart));

    for (int step = 1; step <= t_step; ++step)
    {
        Octree* octree = create_empty_octree(space);

        for (int j = 0; j < npart; ++j)
        {
            if (in_space(space, particle_array[j]))
            {
                octree_insert(octree, space, &particle_array[j]);
            }
        }

        parallel_for_indices(0, npart, nthreads, [&](int idx) {
            particle_array[idx].force_x = 0.0;
            particle_array[idx].force_y = 0.0;
            particle_array[idx].force_z = 0.0;
        });

        parallel_for_indices(0, npart, nthreads, [&](int idx) {
            if (in_space(space, particle_array[idx]))
            {
                compute_force(&particle_array[idx], octree);
            }
        });

        parallel_for_indices(0, npart, nthreads, [&](int idx) {
            if (in_space(space, particle_array[idx]))
            {
                update_particle_position_and_velocity(&particle_array[idx]);
            }
        });

        free_octree(octree);

        print_summary(particle_array.get(), npart, space, step, nthreads);
    }

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart
                << " | Num Steps: " << t_step << "\n";
    }

    return 0;
}
