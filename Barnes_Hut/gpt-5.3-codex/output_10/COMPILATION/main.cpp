/* ============================================================================
 * Barnes-Hut main
 *
 * Usage:
 *   ./barnes-hut --nparticles <NUM> --nsteps <NUM> --hpx:threads=<NUM>
 * ============================================================================ */
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "octree.hpp"

constexpr double WIDTH   = 99999.0;
constexpr double LENGTH  = 99999.0;
constexpr double HEIGHT  = 99999.0;

constexpr double ORIGIN_X = -99999.0;
constexpr double ORIGIN_Y = -99999.0;
constexpr double ORIGIN_Z = -99999.0;

template <typename F>
void parallel_for_chunks(int begin, int end, std::size_t num_tasks, F& f)
{
    if (begin >= end) return;
    if (num_tasks == 0) num_tasks = 1;

    const std::size_t total = static_cast<std::size_t>(end - begin);
    const std::size_t tasks = std::min(num_tasks, total);
    const std::size_t chunk = (total + tasks - 1) / tasks;

    std::vector<std::thread> workers;
    workers.reserve(tasks);

    for (std::size_t t = 0; t < tasks; ++t) {
        const int start = begin + static_cast<int>(t * chunk);
        if (start >= end) break;
        const int stop = std::min(end, start + static_cast<int>(chunk));

        workers.emplace_back([start, stop, &f]() {
            for (int i = start; i < stop; ++i) {
                f(i);
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }
}

static bool starts_with(const std::string& s, const std::string& prefix)
{
    return s.rfind(prefix, 0) == 0;
}

int main(int argc, char* argv[])
{
    int npart = 0;
    int t_step = 0;
    std::size_t requested_threads = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        try {
            if (arg == "--nparticles") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --nparticles");
                npart = std::stoi(argv[++i]);
            } else if (starts_with(arg, "--nparticles=")) {
                npart = std::stoi(arg.substr(std::string("--nparticles=").size()));
            } else if (arg == "--nsteps") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --nsteps");
                t_step = std::stoi(argv[++i]);
            } else if (starts_with(arg, "--nsteps=")) {
                t_step = std::stoi(arg.substr(std::string("--nsteps=").size()));
            } else if (arg == "--hpx:threads") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --hpx:threads");
                requested_threads = static_cast<std::size_t>(std::stoul(argv[++i]));
            } else if (starts_with(arg, "--hpx:threads=")) {
                requested_threads = static_cast<std::size_t>(
                    std::stoul(arg.substr(std::string("--hpx:threads=").size())));
            }
        } catch (std::exception const&) {
            std::cerr << "Usage: ./barnes-hut --nparticles <NUM> --nsteps <NUM> "
                         "[--hpx:threads=<NUM>]\n";
            return 1;
        }
    }

    if (npart <= 0 || t_step <= 0) {
        std::cerr << "Usage: ./barnes-hut --nparticles <NUM> --nsteps <NUM> "
                     "[--hpx:threads=<NUM>]\n";
        return 1;
    }

    // Default simulation space
    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    // Reset output file
    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
    }

    std::vector<Particle> particle_array(static_cast<std::size_t>(npart));
    srand(42);
    generate_random_particles(particle_array.data(), space, npart);

    std::size_t num_tasks = requested_threads;
    if (num_tasks == 0) {
        num_tasks = static_cast<std::size_t>(std::thread::hardware_concurrency());
    }
    if (num_tasks == 0) num_tasks = 1;

    // Main timestep loop
    for (int step = 1; step <= t_step; ++step) {
        // Build octree from current particle positions
        Octree* octree = create_empty_octree(space);
        for (int j = 0; j < npart; ++j) {
            if (in_space(space, particle_array[static_cast<std::size_t>(j)])) {
                octree_insert(octree, space, &particle_array[static_cast<std::size_t>(j)]);
            }
        }

        // PHASE 0: Reset forces
        auto reset_forces = [&](int idx) {
            auto& p = particle_array[static_cast<std::size_t>(idx)];
            p.force_x = 0.0;
            p.force_y = 0.0;
            p.force_z = 0.0;
        };
        parallel_for_chunks(0, npart, num_tasks, reset_forces);

        // PHASE 1: Compute forces
        auto compute_forces = [&](int idx) {
            Particle* p = &particle_array[static_cast<std::size_t>(idx)];
            if (in_space(space, *p)) {
                compute_force(p, octree);
            }
        };
        parallel_for_chunks(0, npart, num_tasks, compute_forces);

        // PHASE 2: Integrate positions/velocities
        auto integrate = [&](int idx) {
            Particle* p = &particle_array[static_cast<std::size_t>(idx)];
            if (in_space(space, *p)) {
                update_particle_position_and_velocity(p);
            }
        };
        parallel_for_chunks(0, npart, num_tasks, integrate);

        free_octree(octree);

        // Print global statistics per step
        print_summary(particle_array.data(), npart, space, step, 1);
    }

    // Final summary line
    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart << " | Num Steps: " << t_step << "\n";
    }

    return 0;
}
