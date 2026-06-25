/* ============================================================================
 * Barnes-Hut main
 * ============================================================================
 */
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "octree.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

constexpr double WIDTH = 99999.0;
constexpr double LENGTH = 99999.0;
constexpr double HEIGHT = 99999.0;

constexpr double ORIGIN_X = -99999.0;
constexpr double ORIGIN_Y = -99999.0;
constexpr double ORIGIN_Z = -99999.0;

namespace {

void print_usage()
{
    std::cout << "Usage: ./barnes-hut --nparticles <NUM> --nsteps <NUM>\n";
}

bool starts_with(std::string const& s, std::string const& prefix)
{
    return s.rfind(prefix, 0) == 0;
}

bool parse_int_arg(std::string const& text, int& out)
{
    char* end = nullptr;
    long v = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || v < INT_MIN || v > INT_MAX) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    int npart = -1;
    int t_step = -1;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            print_usage();
            return 0;
        }

        if (arg == "--nparticles")
        {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], npart))
            {
                print_usage();
                return 1;
            }
            continue;
        }

        if (starts_with(arg, "--nparticles="))
        {
            if (!parse_int_arg(arg.substr(std::string("--nparticles=").size()), npart))
            {
                print_usage();
                return 1;
            }
            continue;
        }

        if (arg == "--nsteps")
        {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], t_step))
            {
                print_usage();
                return 1;
            }
            continue;
        }

        if (starts_with(arg, "--nsteps="))
        {
            if (!parse_int_arg(arg.substr(std::string("--nsteps=").size()), t_step))
            {
                print_usage();
                return 1;
            }
            continue;
        }

        // Ignore legacy HPX runtime flags if present.
        if (starts_with(arg, "--hpx:"))
        {
            continue;
        }

        if (!arg.empty() && arg[0] == '-')
        {
            print_usage();
            return 1;
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

    std::vector<Particle> particle_array(static_cast<std::size_t>(npart));

    std::srand(42);
    generate_random_particles(particle_array.data(), space, npart);

    for (int step = 1; step <= t_step; ++step)
    {
        Octree* octree = create_empty_octree(space);

        for (int j = 0; j < npart; ++j)
        {
            if (in_space(space, particle_array[static_cast<std::size_t>(j)]))
            {
                octree_insert(octree, space, &particle_array[static_cast<std::size_t>(j)]);
            }
        }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int idx = 0; idx < npart; ++idx)
        {
            particle_array[static_cast<std::size_t>(idx)].force_x = 0.0;
            particle_array[static_cast<std::size_t>(idx)].force_y = 0.0;
            particle_array[static_cast<std::size_t>(idx)].force_z = 0.0;
        }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int idx = 0; idx < npart; ++idx)
        {
            if (in_space(space, particle_array[static_cast<std::size_t>(idx)]))
            {
                compute_force(&particle_array[static_cast<std::size_t>(idx)], octree);
            }
        }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int idx = 0; idx < npart; ++idx)
        {
            if (in_space(space, particle_array[static_cast<std::size_t>(idx)]))
            {
                update_particle_position_and_velocity(&particle_array[static_cast<std::size_t>(idx)]);
            }
        }

        free_octree(octree);

        print_summary(particle_array.data(), npart, space, step, 1);
    }

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart << " | Num Steps: " << t_step << "\n";
    }

    return 0;
}
