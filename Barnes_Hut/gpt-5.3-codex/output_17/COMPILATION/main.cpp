/* ============================================================================
 * Barnes-Hut main.cpp
 * Usage:
 *   ./barnes-hut --nparticles <NUM> --nsteps <NUM> [--hpx:threads=<NUM>]
 * ============================================================================ */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#include "octree.hpp"

constexpr double WIDTH   = 100.0;
constexpr double LENGTH  = 100.0;
constexpr double HEIGHT  = 100.0;

constexpr double ORIGIN_X = 0.0;
constexpr double ORIGIN_Y = 0.0;
constexpr double ORIGIN_Z = 0.0;

template <typename F>
void parallel_for(std::size_t begin, std::size_t end, unsigned num_threads, F&& fn)
{
    if (end <= begin) return;

    std::size_t total = end - begin;
    if (num_threads <= 1 || total < 2048) {
        for (std::size_t i = begin; i < end; ++i) {
            fn(i);
        }
        return;
    }

    std::size_t threads = std::min<std::size_t>(num_threads, total);
    std::size_t chunk = (total + threads - 1) / threads;

    std::vector<std::thread> workers;
    workers.reserve(threads - 1);

    std::size_t start = begin;
    for (std::size_t t = 0; t + 1 < threads; ++t) {
        std::size_t s = start;
        std::size_t e = std::min(end, s + chunk);
        workers.emplace_back([s, e, &fn]() {
            for (std::size_t i = s; i < e; ++i) {
                fn(i);
            }
        });
        start = e;
    }

    for (std::size_t i = start; i < end; ++i) {
        fn(i);
    }

    for (auto& th : workers) {
        th.join();
    }
}

int main(int argc, char* argv[])
{
    int npart = 100000;
    int t_step = 20;

    unsigned num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;

    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--nparticles") == 0 && i + 1 < argc) {
            npart = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nsteps") == 0 && i + 1 < argc) {
            t_step = std::atoi(argv[++i]);
        } else if (std::strncmp(argv[i], "--nparticles=", 13) == 0) {
            npart = std::atoi(argv[i] + 13);
        } else if (std::strncmp(argv[i], "--nsteps=", 9) == 0) {
            t_step = std::atoi(argv[i] + 9);
        } else if (std::strncmp(argv[i], "--hpx:threads=", 14) == 0) {
            int parsed = std::atoi(argv[i] + 14);
            if (parsed > 0) num_threads = static_cast<unsigned>(parsed);
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            int parsed = std::atoi(argv[++i]);
            if (parsed > 0) num_threads = static_cast<unsigned>(parsed);
        } else if (std::strncmp(argv[i], "--threads=", 10) == 0) {
            int parsed = std::atoi(argv[i] + 10);
            if (parsed > 0) num_threads = static_cast<unsigned>(parsed);
        }
    }

    if (npart <= 0 || t_step <= 0) {
        std::fprintf(stderr, "Usage: %s --nparticles <NUM> --nsteps <NUM>\n", argv[0]);
        return 1;
    }

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
    }

    auto* particle_array = new Particle[npart]();

    std::srand(42);
    generate_random_particles(particle_array, space, npart);

    for (int step = 0; step < t_step; ++step) {
        Octree* octree = create_empty_octree(space);
        for (int j = 0; j < npart; ++j) {
            if (in_space(space, particle_array[j])) {
                octree_insert(octree, space, &(particle_array[j]));
            }
        }

        parallel_for(0, static_cast<std::size_t>(npart), num_threads, [&](std::size_t idx) {
            particle_array[idx].force_x = 0.0;
            particle_array[idx].force_y = 0.0;
            particle_array[idx].force_z = 0.0;
        });

        parallel_for(0, static_cast<std::size_t>(npart), num_threads, [&](std::size_t idx) {
            Particle* tmp_p = &(particle_array[idx]);
            if (in_space(space, *tmp_p)) {
                compute_force(tmp_p, octree);
            }
        });

        parallel_for(0, static_cast<std::size_t>(npart), num_threads, [&](std::size_t idx) {
            Particle* tmp_p = &(particle_array[idx]);
            if (in_space(space, *tmp_p)) {
                update_particle_position_and_velocity(tmp_p);
            }
        });

        free_octree(octree);

        print_summary(particle_array, npart, space, step + 1, 1);
    }

    delete[] particle_array;

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart << " | Num Steps: " << t_step << "\n";
    }

    return 0;
}
