/* ============================================================================
 * Barnes-Hut Simulation
 *
 * Usage:
 *   ./barnes-hut --nparticles <NUM> --nsteps <NUM> [--hpx:threads=<NUM>]
 * ============================================================================ */
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "octree.hpp"

constexpr double WIDTH   = 99999.0;
constexpr double LENGTH  = 99999.0;
constexpr double HEIGHT  = 99999.0;

constexpr double ORIGIN_X = -99999.0;
constexpr double ORIGIN_Y = -99999.0;
constexpr double ORIGIN_Z = -99999.0;

static bool parse_prefixed_int(char const* arg, char const* key, int& out)
{
    std::size_t key_len = std::strlen(key);
    if (std::strncmp(arg, key, key_len) == 0 && arg[key_len] == '=')
    {
        out = std::atoi(arg + key_len + 1);
        return true;
    }
    return false;
}

int main(int argc, char* argv[])
{
    // Defaults aligned with validation workload
    int npart = 100000;
    int t_step = 20;

    // ------------------------------------------------------------------------
    // Process user input (flag-based parsing)
    // ------------------------------------------------------------------------
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--nparticles") == 0 && i + 1 < argc)
        {
            npart = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--nsteps") == 0 && i + 1 < argc)
        {
            t_step = std::atoi(argv[++i]);
        }
        else if (parse_prefixed_int(argv[i], "--nparticles", npart))
        {
            // handled
        }
        else if (parse_prefixed_int(argv[i], "--nsteps", t_step))
        {
            // handled
        }
        // Ignore unknown options (including --hpx:* runtime flags).
    }

    if (npart <= 0 || t_step <= 0)
    {
        std::cerr << "Usage: " << argv[0]
                  << " --nparticles <NUM> --nsteps <NUM> [--hpx:threads=<NUM>]\n";
        return 1;
    }

    // Simulation domain
    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    // Reset output file
    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
    }

    std::vector<Particle> particle_array(static_cast<std::size_t>(npart));

    // Generate initial particles once
    std::srand(42);
    generate_random_particles(particle_array.data(), space, npart);

    // ------------------------------------------------------------------------
    // Main timestep loop
    // ------------------------------------------------------------------------
    for (int step = 1; step <= t_step; ++step)
    {
        // Build tree
        Octree* octree = create_empty_octree(space);
        for (int j = 0; j < npart; ++j)
        {
            if (in_space(space, particle_array[static_cast<std::size_t>(j)]))
            {
                octree_insert(octree, space, &particle_array[static_cast<std::size_t>(j)]);
            }
        }

        // PHASE 0: RESET FORCES
        for (Particle& p : particle_array)
        {
            p.force_x = 0.0;
            p.force_y = 0.0;
            p.force_z = 0.0;
        }

        // PHASE 1: COMPUTE FORCES
        for (Particle& p : particle_array)
        {
            if (in_space(space, p))
            {
                compute_force(&p, octree);
            }
        }

        // PHASE 2: UPDATE INTEGRATION
        for (Particle& p : particle_array)
        {
            if (in_space(space, p))
            {
                update_particle_position_and_velocity(&p);
            }
        }

        free_octree(octree);

        // Print per-step summary
        print_summary(particle_array.data(), npart, space, step, 1);
    }

    // Final summary line
    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart
                << " | Num Steps: " << t_step << "\n";
    }

    return 0;
}
