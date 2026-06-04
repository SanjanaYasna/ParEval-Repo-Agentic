/* ============================================================================
 * Barnes-Hut N-body simulation with OpenMP parallelism.
 * Original: https://github.com/Jrokisky/MPI-Barnes-hut, but restructured
 * Author:   Justin Rokisky
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
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "octree.hpp"

constexpr double WIDTH   = 99999.0;
constexpr double LENGTH  = 99999.0;
constexpr double HEIGHT  = 99999.0;

constexpr double ORIGIN_X = -99999.0;
constexpr double ORIGIN_Y = -99999.0;
constexpr double ORIGIN_Z = -99999.0;

int main(int argc, char *argv[]) {
    int npart = 0, t_step = 0;

    // ------------------------------------------------------------------------
    // Process user input
    // ------------------------------------------------------------------------
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--nparticles") == 0 && i + 1 < argc) {
            npart = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nsteps") == 0 && i + 1 < argc) {
            t_step = std::atoi(argv[++i]);
        }
    }

    if (npart <= 0 || t_step <= 0) {
        std::fprintf(stderr,
            "Usage: ./barnes-hut --nparticles <NUM> --nsteps <NUM>\n");
        return 1;
    }

    int num_threads = 1;
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#endif

    // Default space.
    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    // Truncate output file.
    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
        outfile.close();
    }

    auto *particle_array = new Particle[npart]();

    // Generate random particles (deterministic, same seed as MPI version).
    srand(42);
    generate_random_particles(particle_array, space, npart);

    // Main timestep loop
    for (int i = t_step; i > 0; i--) {
        // Build the full octree sequentially (tree insertion is inherently
        // sequential due to pointer-based structural mutation).
        Octree *octree = create_empty_octree(space);
        for (int j = 0; j < npart; j++) {
            if (in_space(space, particle_array[j])) {
                octree_insert(octree, space, &(particle_array[j]));
            }
        }

        // --------------------------------------------------------------------
        // PARALLEL PHASES 0 + 1 + 2:
        //   RESET FORCES, COMPUTE FORCES, UPDATE POSITIONS
        // --------------------------------------------------------------------
        #pragma omp parallel for schedule(dynamic, 100)
        for (int idx = 0; idx < npart; idx++) {
            // Phase 0: Reset forces.
            particle_array[idx].force_x = 0.0;
            particle_array[idx].force_y = 0.0;
            particle_array[idx].force_z = 0.0;

            Particle *tmp_p = &(particle_array[idx]);
            if (in_space(space, particle_array[idx])) {
                // Phase 1: Compute forces (read octree, write own forces).
                compute_force(tmp_p, octree);

                // Phase 2: Update position and velocity.
                update_particle_position_and_velocity(tmp_p);
            }
        }

        // No longer need the octree.
        free_octree(octree);

        // Print net particle statistics for this step.
        print_summary(particle_array, npart, space,
                      t_step - i + 1, num_threads);
    }

    // Free particles.
    delete[] particle_array;

    // Append final summary line.
    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart
                << " | Num Steps: " << t_step << "\n";
        outfile.close();
    }

    return 0;
}
