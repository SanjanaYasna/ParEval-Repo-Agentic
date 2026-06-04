/* ============================================================================
 * C++17 threads translation of MPI Barnes-Hut
 * Original: https://github.com/Jrokisky/MPI-Barnes-hut, but restructured
 * Author:   Justin Rokisky
 * Usage:
 *  ./barnes-hut --nparticles <NUM> --nsteps <NUM> [--threads <NUM>]
 * ============================================================================ */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <future>

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
    int num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads <= 0) num_threads = 1;

    // ---- simple CLI parsing for --nparticles, --nsteps, --threads ----
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--nparticles" && i + 1 < argc) {
            npart = std::atoi(argv[++i]);
        } else if (arg == "--nsteps" && i + 1 < argc) {
            t_step = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
            if (num_threads <= 0) num_threads = 1;
        }
    }

    if (npart <= 0 || t_step <= 0) {
        std::fprintf(stderr,
            "Usage: ./barnes-hut --nparticles <NUM> --nsteps <NUM> "
            "[--threads <NUM>]\n");
        return 1;
    }

    // Default space.
    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    // Clear the output file.
    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
        outfile.close();
    }

    auto *particle_array = new Particle[npart]();

    // Generate random particles (deterministic seed).
    srand(42);
    generate_random_particles(particle_array, space, npart);

    // Main timestep loop
    for (int i = t_step; i > 0; i--) {

        // Each timestep: rebuild the octree (sequential – shared structure).
        Octree *octree = create_empty_octree(space);
        for (int j = 0; j < npart; j++) {
            if (in_space(space, particle_array[j])) {
                octree_insert(octree, space, &(particle_array[j]));
            }
        }

        // -----------------------------------------------------------------
        // Partition particles across worker threads, mirroring the
        // MPI rank-based chunking.
        // -----------------------------------------------------------------
        int chunk = npart / num_threads;

        std::vector<std::future<void>> futures;
        futures.reserve(num_threads);

        for (int t = 0; t < num_threads; t++) {
            int startpos = t * chunk;
            int endpos   = (t == num_threads - 1) ? npart
                                                   : startpos + chunk;

            // Launch an async task for each chunk.
            futures.push_back(std::async(std::launch::async,
                [&particle_array, &space, octree, startpos, endpos]()
            {
                // --------------------------------------------------------
                // PHASE 0: RESET FORCES
                // --------------------------------------------------------
                for (int idx = startpos; idx < endpos; idx++) {
                    particle_array[idx].force_x = 0.0;
                    particle_array[idx].force_y = 0.0;
                    particle_array[idx].force_z = 0.0;
                }

                // --------------------------------------------------------
                // PHASE 1: COMPUTE FORCES (read octree, write own forces)
                // --------------------------------------------------------
                for (int idx = startpos; idx < endpos; idx++) {
                    Particle *tmp_p = &(particle_array[idx]);
                    if (in_space(space, particle_array[idx])) {
                        compute_force(tmp_p, octree);
                    }
                }

                // --------------------------------------------------------
                // PHASE 2: UPDATE POSITIONS & VELOCITIES
                // --------------------------------------------------------
                for (int idx = startpos; idx < endpos; idx++) {
                    Particle *tmp_p = &(particle_array[idx]);
                    if (in_space(space, particle_array[idx])) {
                        update_particle_position_and_velocity(tmp_p);
                    }
                }
            }));
        }

        // Synchronise – all threads wrote disjoint slices, so after
        // wait the full particle_array is coherent.
        for (auto &f : futures) {
            f.get();
        }

        // No longer need the octree.
        free_octree(octree);

        // Print summary (sequential).
        print_summary(particle_array, npart, space,
                      t_step - i + 1, num_threads);
    }

    // Free particles.
    delete[] particle_array;

    // Append final line.
    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart
                << " | Num Steps: " << t_step << "\n";
        outfile.close();
    }

    return 0;
}
