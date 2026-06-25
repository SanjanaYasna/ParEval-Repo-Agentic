/* ============================================================================
 * Barnes-Hut main
 * Usage:
 *   ./barnes-hut --nparticles <NUM> --nsteps <NUM>
 * ============================================================================ */
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "octree.hpp"

constexpr double WIDTH = 99999.0;
constexpr double LENGTH = 99999.0;
constexpr double HEIGHT = 99999.0;

constexpr double ORIGIN_X = -99999.0;
constexpr double ORIGIN_Y = -99999.0;
constexpr double ORIGIN_Z = -99999.0;

namespace
{
bool parse_int_value(const char* s, int& out)
{
    if (s == nullptr || *s == '\0')
    {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0')
    {
        return false;
    }
    if (v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max())
    {
        return false;
    }

    out = static_cast<int>(v);
    return true;
}

void print_usage()
{
    std::cerr << "Usage: ./barnes-hut --nparticles <NUM> --nsteps <NUM>\n";
}
}    // namespace

int main(int argc, char* argv[])
{
    int npart = 0;
    int t_step = 0;
    bool parse_error = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);

        if (arg == "--help" || arg == "-h")
        {
            print_usage();
            return 0;
        }
        else if (arg.rfind("--nparticles=", 0) == 0)
        {
            const std::string val = arg.substr(std::string("--nparticles=").size());
            if (!parse_int_value(val.c_str(), npart))
            {
                parse_error = true;
            }
        }
        else if (arg == "--nparticles")
        {
            if (i + 1 >= argc || !parse_int_value(argv[++i], npart))
            {
                parse_error = true;
            }
        }
        else if (arg.rfind("--nsteps=", 0) == 0)
        {
            const std::string val = arg.substr(std::string("--nsteps=").size());
            if (!parse_int_value(val.c_str(), t_step))
            {
                parse_error = true;
            }
        }
        else if (arg == "--nsteps")
        {
            if (i + 1 >= argc || !parse_int_value(argv[++i], t_step))
            {
                parse_error = true;
            }
        }
        else if (arg.rfind("--hpx:", 0) == 0)
        {
            // Ignore HPX-style options for compatibility.
            if (arg.find('=') == std::string::npos && i + 1 < argc)
            {
                std::string next(argv[i + 1]);
                if (next.rfind("--", 0) != 0)
                {
                    ++i;
                }
            }
        }
    }

    if (parse_error || npart <= 0 || t_step <= 0)
    {
        print_usage();
        return 1;
    }

    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
    }

    auto* particle_array = new Particle[npart]();
    generate_random_particles(particle_array, space, npart);

    for (int step = 1; step <= t_step; ++step)
    {
        Octree* octree = create_empty_octree(space);
        for (int j = 0; j < npart; ++j)
        {
            if (in_space(space, particle_array[j]))
            {
                octree_insert(octree, space, &(particle_array[j]));
            }
        }

        for (int idx = 0; idx < npart; ++idx)
        {
            particle_array[idx].force_x = 0.0;
            particle_array[idx].force_y = 0.0;
            particle_array[idx].force_z = 0.0;
        }

        for (int idx = 0; idx < npart; ++idx)
        {
            if (in_space(space, particle_array[idx]))
            {
                compute_force(&(particle_array[idx]), octree);
            }
        }

        for (int idx = 0; idx < npart; ++idx)
        {
            if (in_space(space, particle_array[idx]))
            {
                update_particle_position_and_velocity(&(particle_array[idx]));
            }
        }

        free_octree(octree);
        print_summary(particle_array, npart, space, step, 1);
    }

    delete[] particle_array;

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart << " | Num Steps: " << t_step
                << "\n";
    }

    return 0;
}
