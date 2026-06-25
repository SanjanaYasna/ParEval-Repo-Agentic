/* ============================================================================
 * Barnes-Hut main
 * ============================================================================
 */
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "octree.hpp"

constexpr double WIDTH = 100.0;
constexpr double LENGTH = 100.0;
constexpr double HEIGHT = 100.0;

constexpr double ORIGIN_X = 0.0;
constexpr double ORIGIN_Y = 0.0;
constexpr double ORIGIN_Z = 0.0;

static void print_usage()
{
    std::cerr << "Usage: ./barnes-hut [--nparticles <NUM>] [--nsteps <NUM>] [--nprocs <NUM>]\n";
}

static bool parse_int(const std::string& s, int& out)
{
    try {
        std::size_t pos = 0;
        long v = std::stol(s, &pos);
        if (pos != s.size()) {
            return false;
        }
        out = static_cast<int>(v);
        return true;
    } catch (...) {
        return false;
    }
}

int main(int argc, char* argv[])
{
    int npart = 100000;
    int t_step = 20;
    bool help = false;
    bool parse_error = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            help = true;
            continue;
        }

        if (arg.rfind("--hpx:", 0) == 0) {
            continue;
        }

        if (arg.rfind("--nparticles=", 0) == 0) {
            if (!parse_int(arg.substr(std::string("--nparticles=").size()), npart)) {
                parse_error = true;
            }
            continue;
        }

        if (arg == "--nparticles") {
            if (i + 1 < argc && parse_int(argv[i + 1], npart)) {
                ++i;
            } else {
                parse_error = true;
            }
            continue;
        }

        if (arg.rfind("--nsteps=", 0) == 0) {
            if (!parse_int(arg.substr(std::string("--nsteps=").size()), t_step)) {
                parse_error = true;
            }
            continue;
        }

        if (arg == "--nsteps") {
            if (i + 1 < argc && parse_int(argv[i + 1], t_step)) {
                ++i;
            } else {
                parse_error = true;
            }
            continue;
        }

        if (arg == "--nprocs" || arg.rfind("--nprocs=", 0) == 0) {
            if (arg == "--nprocs" && i + 1 < argc) {
                ++i; // compatibility option, ignored
            }
            continue;
        }
    }

    if (help || parse_error || npart <= 0 || t_step <= 0) {
        print_usage();
        return (help && !parse_error) ? 0 : 1;
    }

    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
    }

    std::vector<Particle> particle_array(static_cast<std::size_t>(npart));
    std::srand(42);
    generate_random_particles(particle_array.data(), space, npart);

    for (int step = 1; step <= t_step; ++step) {
        Octree* octree = create_empty_octree(space);

        for (int j = 0; j < npart; ++j) {
            if (in_space(space, particle_array[j])) {
                octree_insert(octree, space, &particle_array[j]);
            }
        }

        for (std::size_t idx = 0; idx < particle_array.size(); ++idx) {
            particle_array[idx].force_x = 0.0;
            particle_array[idx].force_y = 0.0;
            particle_array[idx].force_z = 0.0;
        }

        for (std::size_t idx = 0; idx < particle_array.size(); ++idx) {
            Particle* tmp_p = &particle_array[idx];
            if (in_space(space, particle_array[idx])) {
                compute_force(tmp_p, octree);
            }
        }

        for (std::size_t idx = 0; idx < particle_array.size(); ++idx) {
            Particle* tmp_p = &particle_array[idx];
            if (in_space(space, particle_array[idx])) {
                update_particle_position_and_velocity(tmp_p);
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
