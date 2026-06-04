/* ============================================================================
 * C++ threads translation of MPI Barnes-Hut
 * Original: https://github.com/Jrokisky/MPI-Barnes-hut
 * Usage:
 *  ./barnes-hut --nparticles <NUM> --nsteps <NUM>
 * ============================================================================ */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <future>
#include <thread>
#include <string>

#include "octree.hpp"

constexpr double WIDTH   = 99999.0;
constexpr double LENGTH  = 99999.0;
constexpr double HEIGHT  = 99999.0;

constexpr double ORIGIN_X = -99999.0;
constexpr double ORIGIN_Y = -99999.0;
constexpr double ORIGIN_Z = -99999.0;

int main(int argc, char *argv[])
{
    int npart  = 0;
    int t_step = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--nparticles" && i + 1 < argc) {
            npart = std::atoi(argv[++i]);
        } else if (arg == "--nsteps" && i + 1 < argc) {
            t_step = std::atoi(argv[++i]);
        }
    }

    if (npart <= 0 || t_step <= 0) {
        std::cerr << "Usage: barnes-hut --nparticles <NUM> --nsteps <NUM>\n";
        return 1;
    }

    // Default space.
    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    // Clear output file.
    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
        outfile.close();
    }

    auto *particle_array = new Particle[npart]();

    // Generate random particles (deterministic, seed 42).
    srand(42);
    generate_random_particles(particle_array, space, npart);

    // Number of hardware threads available.
    std::size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;

    // Main timestep loop
    for (int i = t_step; i > 0; i--) {
        // Each step rebuilds the octree (sequential – tree insertion is
        // inherently serial in this implementation).
        Octree *octree = create_empty_octree(space);
        for (int j = 0; j < npart; j++) {
            if (in_space(space, particle_array[j])) {
                octree_insert(octree, space, &(particle_array[j]));
            }
        }

        // ----------------------------------------------------------------
        // Distribute particle work across threads, mirroring the
        // MPI per-rank chunk split.  Each task owns a disjoint slice of
        // the particle array, so no data races occur.
        // ----------------------------------------------------------------
        std::size_t actual_workers =
            std::min(num_threads, static_cast<std::size_t>(npart));
        int chunk = npart / static_cast<int>(actual_workers);

        std::vector<std::future<void>> futures;
        futures.reserve(actual_workers);

        for (std::size_t t = 0; t < actual_workers; t++) {
            int startpos = static_cast<int>(t) * chunk;
            int endpos   = (t == actual_workers - 1)
                               ? npart
                               : static_cast<int>(t + 1) * chunk;

            futures.push_back(std::async(std::launch::async,
                [particle_array, space, octree, startpos, endpos]() {
                    // ------------------------------------------------
                    // PHASE 0: RESET FORCES
                    // ------------------------------------------------
                    for (int idx = startpos; idx < endpos; idx++) {
                        particle_array[idx].force_x = 0.0;
                        particle_array[idx].force_y = 0.0;
                        particle_array[idx].force_z = 0.0;
                    }

                    // ------------------------------------------------
                    // PHASE 1: COMPUTE FORCES (read tree, write forces)
                    // ------------------------------------------------
                    for (int idx = startpos; idx < endpos; idx++) {
                        Particle *tmp_p = &(particle_array[idx]);
                        if (in_space(space, particle_array[idx])) {
                            compute_force(tmp_p, octree);
                        }
                    }

                    // ------------------------------------------------
                    // PHASE 2: UPDATE POSITIONS & VELOCITIES
                    // ------------------------------------------------
                    for (int idx = startpos; idx < endpos; idx++) {
                        Particle *tmp_p = &(particle_array[idx]);
                        if (in_space(space, particle_array[idx])) {
                            update_particle_position_and_velocity(tmp_p);
                        }
                    }
                }));
        }

        // Synchronise all tasks.
        for (auto &f : futures) {
            f.get();
        }

        // No longer need the octree.
        free_octree(octree);

        // Print per-step summary.
        print_summary(particle_array, npart, space,
                      t_step - i + 1, static_cast<int>(num_threads));
    }

    // Free particles.
    delete[] particle_array;

    // Final summary line.
    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart
                << " | Num Steps: " << t_step << "\n";
        outfile.close();
    }

    return 0;
}
