/* ============================================================================
 * Barnes-Hut Simulation
 * Original: https://github.com/Jrokisky/MPI-Barnes-hut
 * ============================================================================ */
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "octree.hpp"

constexpr double WIDTH   = 99999.0;
constexpr double LENGTH  = 99999.0;
constexpr double HEIGHT  = 99999.0;

constexpr double ORIGIN_X = -99999.0;
constexpr double ORIGIN_Y = -99999.0;
constexpr double ORIGIN_Z = -99999.0;

static bool parse_int(const std::string& s, int& out)
{
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') {
        return false;
    }
    if (v < static_cast<long>(std::numeric_limits<int>::min()) ||
        v > static_cast<long>(std::numeric_limits<int>::max())) {
        return false;
    }
    out = static_cast<int>(v);
    return true;
}

static bool parse_prefixed_int(const std::string& arg, const std::string& key, int& out)
{
    std::string prefix = key + "=";
    if (arg.rfind(prefix, 0) != 0) {
        return false;
    }
    return parse_int(arg.substr(prefix.size()), out);
}

int main(int argc, char* argv[])
{
    int npart = 0;
    int t_step = 0;
    int nthreads = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--nparticles" && (i + 1) < argc) {
            parse_int(argv[++i], npart);
        } else if (arg == "--nsteps" && (i + 1) < argc) {
            parse_int(argv[++i], t_step);
        } else if (arg == "--nprocs" && (i + 1) < argc) {
            int ignored = 0;
            parse_int(argv[++i], ignored);
        } else if (arg == "--nthreads" && (i + 1) < argc) {
            parse_int(argv[++i], nthreads);
        } else if (arg == "--hpx:threads" && (i + 1) < argc) {
            parse_int(argv[++i], nthreads);
        } else if (parse_prefixed_int(arg, "--nparticles", npart)) {
            continue;
        } else if (parse_prefixed_int(arg, "--nsteps", t_step)) {
            continue;
        } else if (parse_prefixed_int(arg, "--nthreads", nthreads)) {
            continue;
        } else if (parse_prefixed_int(arg, "--hpx:threads", nthreads)) {
            continue;
        }
    }

    if (npart <= 0 || t_step <= 0) {
        std::fprintf(stderr,
            "Usage: ./barnes-hut --nparticles <NUM> --nsteps <NUM> [--nthreads=<NUM>]\n");
        return 0;
    }

    if (nthreads > 0) {
        set_num_threads(static_cast<unsigned int>(nthreads));
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

        parallel_for(0, npart, [&](int idx) {
            particle_array[idx].force_x = 0.0;
            particle_array[idx].force_y = 0.0;
            particle_array[idx].force_z = 0.0;
        });

        parallel_for(0, npart, [&](int idx) {
            if (in_space(space, particle_array[idx])) {
                compute_force(&particle_array[idx], octree);
            }
        });

        parallel_for(0, npart, [&](int idx) {
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
