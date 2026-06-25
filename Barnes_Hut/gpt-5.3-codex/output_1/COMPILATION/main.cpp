/* ============================================================================
 * Barnes-Hut main
 * ============================================================================
 */
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "octree.hpp"

constexpr double WIDTH  = 99999.0;
constexpr double LENGTH = 99999.0;
constexpr double HEIGHT = 99999.0;

constexpr double ORIGIN_X = -99999.0;
constexpr double ORIGIN_Y = -99999.0;
constexpr double ORIGIN_Z = -99999.0;

namespace {

void print_usage()
{
    std::fprintf(stderr,
        "Usage: barnes-hut [--nparticles <NUM>] [--nsteps <NUM>] [--hpx:threads=<N>]\n");
}

bool parse_int(std::string const& s, int& out)
{
    try {
        std::size_t idx = 0;
        long v = std::stol(s, &idx, 10);
        if (idx != s.size() || v < INT_MIN || v > INT_MAX) {
            return false;
        }
        out = static_cast<int>(v);
        return true;
    } catch (...) {
        return false;
    }
}

}    // namespace

int main(int argc, char* argv[])
{
    int npart = 100000;
    int t_step = 20;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--nparticles") {
            if (i + 1 >= argc || !parse_int(argv[++i], npart)) {
                print_usage();
                return 1;
            }
        } else if (arg.rfind("--nparticles=", 0) == 0) {
            if (!parse_int(arg.substr(13), npart)) {
                print_usage();
                return 1;
            }
        } else if (arg == "--nsteps") {
            if (i + 1 >= argc || !parse_int(argv[++i], t_step)) {
                print_usage();
                return 1;
            }
        } else if (arg.rfind("--nsteps=", 0) == 0) {
            if (!parse_int(arg.substr(9), t_step)) {
                print_usage();
                return 1;
            }
        } else if (arg.rfind("--hpx:", 0) == 0) {
            if (arg.find('=') == std::string::npos && i + 1 < argc && argv[i + 1][0] != '-') {
                ++i;
            }
            continue;
        } else {
            print_usage();
            return 1;
        }
    }

    if (npart <= 0 || t_step <= 0) {
        print_usage();
        return 1;
    }

    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
    }

    auto* particle_array = new Particle[npart]();
    std::srand(42);
    generate_random_particles(particle_array, space, npart);

    for (int step = 1; step <= t_step; ++step) {
        Octree* octree = create_empty_octree(space);

        for (int j = 0; j < npart; ++j) {
            if (in_space(space, particle_array[j])) {
                octree_insert(octree, space, &(particle_array[j]));
            }
        }

        for (int i = 0; i < npart; ++i) {
            particle_array[i].force_x = 0.0;
            particle_array[i].force_y = 0.0;
            particle_array[i].force_z = 0.0;
        }

        for (int i = 0; i < npart; ++i) {
            if (in_space(space, particle_array[i])) {
                compute_force(&particle_array[i], octree);
            }
        }

        for (int i = 0; i < npart; ++i) {
            if (in_space(space, particle_array[i])) {
                update_particle_position_and_velocity(&particle_array[i]);
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
