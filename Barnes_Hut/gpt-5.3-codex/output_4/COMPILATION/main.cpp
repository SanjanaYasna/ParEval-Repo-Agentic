/* ============================================================================
 * Barnes-Hut simulation
 * Usage:
 *   ./barnes-hut --nparticles <NUM> --nsteps <NUM>
 * ============================================================================ */
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "octree.hpp"

constexpr double WIDTH   = 99999.0;
constexpr double LENGTH  = 99999.0;
constexpr double HEIGHT  = 99999.0;

constexpr double ORIGIN_X = -99999.0;
constexpr double ORIGIN_Y = -99999.0;
constexpr double ORIGIN_Z = -99999.0;

struct ParseResult {
    bool found{false};
    bool valid{true};
};

static ParseResult parse_int_option(int argc, char* argv[], const std::string& name, int& out)
{
    ParseResult result{};
    const std::string key = "--" + name;
    const std::string prefix = key + "=";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == key) {
            result.found = true;
            if (i + 1 >= argc) {
                result.valid = false;
                return result;
            }

            try {
                std::size_t pos = 0;
                const std::string val = argv[++i];
                int parsed = std::stoi(val, &pos);
                if (pos != val.size()) {
                    result.valid = false;
                    return result;
                }
                out = parsed;
            } catch (...) {
                result.valid = false;
                return result;
            }
            return result;
        }

        if (arg.rfind(prefix, 0) == 0) {
            result.found = true;
            try {
                std::size_t pos = 0;
                const std::string val = arg.substr(prefix.size());
                int parsed = std::stoi(val, &pos);
                if (pos != val.size()) {
                    result.valid = false;
                    return result;
                }
                out = parsed;
            } catch (...) {
                result.valid = false;
                return result;
            }
            return result;
        }
    }

    return result;
}

int main(int argc, char* argv[])
{
    int npart = 100000;
    int t_step = 20;

    const ParseResult nparticles_parse = parse_int_option(argc, argv, "nparticles", npart);
    const ParseResult nsteps_parse = parse_int_option(argc, argv, "nsteps", t_step);

    if (!nparticles_parse.valid || !nsteps_parse.valid || npart <= 0 || t_step <= 0) {
        std::cerr << "Usage: ./barnes-hut --nparticles <NUM> --nsteps <NUM>\n";
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
            if (in_space(space, particle_array[static_cast<std::size_t>(j)])) {
                octree_insert(octree, space, &particle_array[static_cast<std::size_t>(j)]);
            }
        }

        for (int idx = 0; idx < npart; ++idx) {
            Particle& p = particle_array[static_cast<std::size_t>(idx)];
            p.force_x = 0.0;
            p.force_y = 0.0;
            p.force_z = 0.0;
        }

        for (int idx = 0; idx < npart; ++idx) {
            Particle* p = &particle_array[static_cast<std::size_t>(idx)];
            if (in_space(space, *p)) {
                compute_force(p, octree);
            }
        }

        for (int idx = 0; idx < npart; ++idx) {
            Particle* p = &particle_array[static_cast<std::size_t>(idx)];
            if (in_space(space, *p)) {
                update_particle_position_and_velocity(p);
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
