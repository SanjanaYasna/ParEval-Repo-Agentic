/* ============================================================================
 * Barnes-Hut main
 * Usage:
 *   ./barnes-hut --nparticles <NUM> --nsteps <NUM>
 * ============================================================================ */
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>

#include "octree.hpp"

constexpr double WIDTH   = 99999.0;
constexpr double LENGTH  = 99999.0;
constexpr double HEIGHT  = 99999.0;

constexpr double ORIGIN_X = -99999.0;
constexpr double ORIGIN_Y = -99999.0;
constexpr double ORIGIN_Z = -99999.0;

template <typename F>
void parallel_for_indices(int begin, int end, F&& f)
{
    if (end <= begin) return;

    const std::size_t n = static_cast<std::size_t>(end - begin);
    std::size_t max_threads = static_cast<std::size_t>(std::thread::hardware_concurrency());
    if (max_threads == 0) {
        max_threads = 4;
    }

    const std::size_t num_tasks = std::min<std::size_t>(n, max_threads);
    const std::size_t block_size = (n + num_tasks - 1) / num_tasks;

    std::vector<std::thread> workers;
    workers.reserve(num_tasks);

    for (std::size_t t = 0; t < num_tasks; ++t) {
        const int b = begin + static_cast<int>(t * block_size);
        const int e = std::min(end, b + static_cast<int>(block_size));
        if (b >= e) break;

        workers.emplace_back([b, e, &f]() {
            for (int i = b; i < e; ++i) {
                f(i);
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }
}

static bool parse_int_str(const std::string& s, int& out)
{
    char* end = nullptr;
    long val = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0') return false;
    if (val < INT_MIN || val > INT_MAX) return false;
    out = static_cast<int>(val);
    return true;
}

int main(int argc, char* argv[])
{
    int npart = 0;
    int t_step = 0;

    bool parse_error = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: barnes-hut --nparticles <NUM> --nsteps <NUM>\n";
            return 0;
        }

        auto parse_option = [&](const std::string& name, int& target) -> bool {
            if (arg == name) {
                if (i + 1 >= argc) {
                    parse_error = true;
                    return true;
                }
                int v = 0;
                if (!parse_int_str(argv[++i], v)) {
                    parse_error = true;
                    return true;
                }
                target = v;
                return true;
            }

            std::string prefix = name + "=";
            if (arg.rfind(prefix, 0) == 0) {
                int v = 0;
                if (!parse_int_str(arg.substr(prefix.size()), v)) {
                    parse_error = true;
                    return true;
                }
                target = v;
                return true;
            }

            return false;
        };

        if (parse_option("--nparticles", npart)) continue;
        if (parse_option("--nsteps", t_step)) continue;

        // Ignore unknown arguments (e.g. legacy runtime options)
    }

    if (parse_error || npart <= 0 || t_step <= 0) {
        std::fprintf(stderr, "Usage: barnes-hut --nparticles <NUM> --nsteps <NUM>\n");
        return 1;
    }

    Space space = {WIDTH, LENGTH, HEIGHT, ORIGIN_X, ORIGIN_Y, ORIGIN_Z};

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::trunc);
    }

    auto* particle_array = new Particle[npart]();

    std::srand(42);
    generate_random_particles(particle_array, space, npart);

    for (int i = t_step; i > 0; --i) {
        Octree* octree = create_empty_octree(space);
        for (int j = 0; j < npart; ++j) {
            if (in_space(space, particle_array[j])) {
                octree_insert(octree, space, &(particle_array[j]));
            }
        }

        parallel_for_indices(0, npart, [&](int idx) {
            particle_array[idx].force_x = 0.0;
            particle_array[idx].force_y = 0.0;
            particle_array[idx].force_z = 0.0;
        });

        parallel_for_indices(0, npart, [&](int idx) {
            Particle* tmp_p = &(particle_array[idx]);
            if (in_space(space, *tmp_p)) {
                compute_force(tmp_p, octree);
            }
        });

        parallel_for_indices(0, npart, [&](int idx) {
            Particle* tmp_p = &(particle_array[idx]);
            if (in_space(space, *tmp_p)) {
                update_particle_position_and_velocity(tmp_p);
            }
        });

        free_octree(octree);

        print_summary(particle_array, npart, space, t_step - i + 1, 1);
    }

    delete[] particle_array;

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart << " | Num Steps: " << t_step << "\n";
    }

    return 0;
}
