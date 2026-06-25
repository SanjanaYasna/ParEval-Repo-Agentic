/* ============================================================================
 * Barnes-Hut Simulation
 * Usage:
 *  ./barnes-hut --nparticles <NUM> --nsteps <NUM> [--hpx:threads=<NUM>]
 * ============================================================================ */
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
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

static unsigned g_num_threads = std::max(1u, std::thread::hardware_concurrency());

bool parse_int(std::string const& s, int& out)
{
    try
    {
        std::size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size())
        {
            return false;
        }
        out = v;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool parse_uint(std::string const& s, unsigned& out)
{
    try
    {
        std::size_t pos = 0;
        unsigned long v = std::stoul(s, &pos);
        if (pos != s.size() || v == 0UL || v > std::numeric_limits<unsigned>::max())
        {
            return false;
        }
        out = static_cast<unsigned>(v);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

void print_usage()
{
    std::fprintf(stderr,
        "Usage: ./barnes-hut --nparticles <NUM> --nsteps <NUM> [--hpx:threads=<NUM>]\n");
}

bool parse_args(int argc, char* argv[], int& npart, int& nsteps)
{
    npart = 0;
    nsteps = 0;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--nparticles")
        {
            if (i + 1 >= argc || !parse_int(argv[++i], npart))
            {
                return false;
            }
        }
        else if (arg.rfind("--nparticles=", 0) == 0)
        {
            if (!parse_int(arg.substr(std::string("--nparticles=").size()), npart))
            {
                return false;
            }
        }
        else if (arg == "--nsteps")
        {
            if (i + 1 >= argc || !parse_int(argv[++i], nsteps))
            {
                return false;
            }
        }
        else if (arg.rfind("--nsteps=", 0) == 0)
        {
            if (!parse_int(arg.substr(std::string("--nsteps=").size()), nsteps))
            {
                return false;
            }
        }
        else if (arg == "--hpx:threads")
        {
            if (i + 1 >= argc || !parse_uint(argv[++i], g_num_threads))
            {
                return false;
            }
        }
        else if (arg.rfind("--hpx:threads=", 0) == 0)
        {
            if (!parse_uint(arg.substr(std::string("--hpx:threads=").size()), g_num_threads))
            {
                return false;
            }
        }
    }

    return (npart > 0 && nsteps > 0);
}

template <typename F>
void parallel_for_chunks(int n, F&& f, std::size_t grain_size = 512)
{
    if (n <= 0)
    {
        return;
    }

    if (grain_size == 0)
    {
        grain_size = 1;
    }

    const std::size_t total_chunks =
        (static_cast<std::size_t>(n) + grain_size - 1) / grain_size;

    unsigned workers = std::min<unsigned>(g_num_threads, static_cast<unsigned>(total_chunks));
    workers = std::max(1u, workers);

    if (workers == 1)
    {
        for (int idx = 0; idx < n; ++idx)
        {
            f(idx);
        }
        return;
    }

    std::atomic<int> next_begin{0};
    std::vector<std::thread> threads;
    threads.reserve(workers);

    for (unsigned t = 0; t < workers; ++t)
    {
        threads.emplace_back([&]() {
            while (true)
            {
                int begin = next_begin.fetch_add(static_cast<int>(grain_size), std::memory_order_relaxed);
                if (begin >= n)
                {
                    break;
                }

                int end = std::min(n, begin + static_cast<int>(grain_size));
                for (int idx = begin; idx < end; ++idx)
                {
                    f(idx);
                }
            }
        });
    }

    for (auto& th : threads)
    {
        th.join();
    }
}

int main(int argc, char* argv[])
{
    int npart = 0;
    int t_step = 0;

    if (!parse_args(argc, argv, npart, t_step))
    {
        print_usage();
        return 1;
    }

    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
    }

    std::vector<Particle> particle_array(static_cast<std::size_t>(npart));

    std::srand(42);
    generate_random_particles(particle_array.data(), space, npart);

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

        parallel_for_chunks(npart, [&](int idx) {
            particle_array[idx].force_x = 0.0;
            particle_array[idx].force_y = 0.0;
            particle_array[idx].force_z = 0.0;
        });

        parallel_for_chunks(npart, [&](int idx) {
            if (in_space(space, particle_array[idx]))
            {
                compute_force(&particle_array[idx], octree);
            }
        });

        parallel_for_chunks(npart, [&](int idx) {
            if (in_space(space, particle_array[idx]))
            {
                update_particle_position_and_velocity(&particle_array[idx]);
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
