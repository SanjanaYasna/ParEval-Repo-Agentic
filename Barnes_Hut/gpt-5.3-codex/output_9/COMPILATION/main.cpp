/* ============================================================================
 * Barnes-Hut main
 *
 * Usage:
 *   ./barnes-hut --nparticles <NUM> --nsteps <NUM> [--hpx:threads=<NUM>]
 * ============================================================================ */
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "octree.hpp"

constexpr double WIDTH = 99999.0;
constexpr double LENGTH = 99999.0;
constexpr double HEIGHT = 99999.0;

constexpr double ORIGIN_X = -99999.0;
constexpr double ORIGIN_Y = -99999.0;
constexpr double ORIGIN_Z = -99999.0;

static bool parse_int(std::string const& s, int& out)
{
    try {
        std::size_t pos = 0;
        int v = std::stoi(s, &pos);
        if (pos != s.size()) {
            return false;
        }
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

static bool read_option_value(int& i, int argc, char* argv[], std::string const& option, std::string& value)
{
    std::string arg = argv[i];
    std::string prefix = option + "=";
    if (arg == option) {
        if (i + 1 >= argc) {
            return false;
        }
        value = argv[++i];
        return true;
    }
    if (arg.rfind(prefix, 0) == 0) {
        value = arg.substr(prefix.size());
        return true;
    }
    return false;
}

int main(int argc, char* argv[])
{
    int npart = 0;
    int t_step = 0;
    int num_threads = 0;
    bool show_help = false;

    for (int i = 1; i < argc; ++i) {
        std::string value;
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            show_help = true;
            continue;
        }

        if (read_option_value(i, argc, argv, "--nparticles", value)) {
            if (!parse_int(value, npart)) {
                std::cerr << "Invalid value for --nparticles\n";
                return 1;
            }
            continue;
        }

        if (read_option_value(i, argc, argv, "--nsteps", value)) {
            if (!parse_int(value, t_step)) {
                std::cerr << "Invalid value for --nsteps\n";
                return 1;
            }
            continue;
        }

        if (read_option_value(i, argc, argv, "--hpx:threads", value) ||
            read_option_value(i, argc, argv, "--threads", value)) {
            if (!parse_int(value, num_threads)) {
                std::cerr << "Invalid value for thread count\n";
                return 1;
            }
            continue;
        }

        if (arg.rfind("--hpx:", 0) == 0) {
            continue;
        }
    }

    if (show_help || npart <= 0 || t_step <= 0) {
        std::cerr << "Usage: barnes-hut --nparticles <NUM> --nsteps <NUM> [--hpx:threads=<NUM>]\n";
        return (show_help ? 0 : 1);
    }

#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#else
    (void) num_threads;
#endif

    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
    }

    auto* particle_array = new Particle[npart]();

    srand(42);
    generate_random_particles(particle_array, space, npart);

    for (int step = 1; step <= t_step; ++step) {
        Octree* octree = create_empty_octree(space);
        for (int j = 0; j < npart; j++) {
            if (in_space(space, particle_array[j])) {
                octree_insert(octree, space, &(particle_array[j]));
            }
        }

#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < npart; ++idx) {
            particle_array[idx].force_x = 0.0;
            particle_array[idx].force_y = 0.0;
            particle_array[idx].force_z = 0.0;
        }

#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < npart; ++idx) {
            if (in_space(space, particle_array[idx])) {
                compute_force(&(particle_array[idx]), octree);
            }
        }

#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < npart; ++idx) {
            if (in_space(space, particle_array[idx])) {
                update_particle_position_and_velocity(&(particle_array[idx]));
            }
        }

        free_octree(octree);

        print_summary(particle_array, npart, space, step, 1);
    }

    delete[] particle_array;

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart << " | Num Steps: " << t_step << "\n";
    }

    return 0;
}
