/* ============================================================================
 * Barnes-Hut main
 * ============================================================================ */
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "octree.hpp"

constexpr double WIDTH = 99999.0;
constexpr double LENGTH = 99999.0;
constexpr double HEIGHT = 99999.0;

constexpr double ORIGIN_X = -99999.0;
constexpr double ORIGIN_Y = -99999.0;
constexpr double ORIGIN_Z = -99999.0;

static void print_usage(std::ostream& os)
{
    os << "Usage: ./barnes-hut --nparticles <NUM> --nsteps <NUM> "
          "[--threads <NUM>]\n";
}

template <typename Func>
inline void parallel_for(std::size_t begin, std::size_t end, int threads, Func const& fn)
{
    if (end <= begin) {
        return;
    }

    std::size_t count = end - begin;
    std::size_t num_threads = threads > 0 ? static_cast<std::size_t>(threads) : 1;
    num_threads = std::min(num_threads, count);

    if (num_threads <= 1 || count < 2048) {
        for (std::size_t i = begin; i < end; ++i) {
            fn(i);
        }
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    std::size_t base = count / num_threads;
    std::size_t rem = count % num_threads;
    std::size_t chunk_begin = begin;

    for (std::size_t t = 0; t < num_threads; ++t) {
        std::size_t len = base + (t < rem ? 1 : 0);
        std::size_t chunk_end = chunk_begin + len;

        workers.emplace_back([chunk_begin, chunk_end, &fn]() {
            for (std::size_t i = chunk_begin; i < chunk_end; ++i) {
                fn(i);
            }
        });

        chunk_begin = chunk_end;
    }

    for (auto& w : workers) {
        w.join();
    }
}

static bool parse_int_strict(std::string const& s, int& out)
{
    try {
        std::size_t pos = 0;
        long long v = std::stoll(s, &pos);
        if (pos != s.size() || v < static_cast<long long>(INT_MIN) ||
            v > static_cast<long long>(INT_MAX)) {
            return false;
        }
        out = static_cast<int>(v);
        return true;
    } catch (...) {
        return false;
    }
}

static bool consume_option_value(int argc, char* argv[], int& i,
                                 std::string const& arg,
                                 std::string const& key,
                                 std::string& value)
{
    if (arg == key) {
        if (i + 1 >= argc) {
            return false;
        }
        value = argv[++i];
        return true;
    }

    std::string prefix = key + "=";
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
    int nthreads = static_cast<int>(std::thread::hardware_concurrency());
    if (nthreads <= 0) {
        nthreads = 1;
    }

    bool show_help = false;
    bool parse_error = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            show_help = true;
            continue;
        }

        std::string value;

        if (consume_option_value(argc, argv, i, arg, "--nparticles", value)) {
            if (!parse_int_strict(value, npart)) {
                parse_error = true;
            }
            continue;
        }

        if (consume_option_value(argc, argv, i, arg, "--nsteps", value)) {
            if (!parse_int_strict(value, t_step)) {
                parse_error = true;
            }
            continue;
        }

        if (consume_option_value(argc, argv, i, arg, "--threads", value)) {
            if (!parse_int_strict(value, nthreads)) {
                parse_error = true;
            }
            continue;
        }

        if (consume_option_value(argc, argv, i, arg, "--hpx:threads", value)) {
            if (value == "all" || value == "cores" || value == "max") {
                int hc = static_cast<int>(std::thread::hardware_concurrency());
                nthreads = hc > 0 ? hc : 1;
            } else {
                int parsed = 0;
                if (parse_int_strict(value, parsed) && parsed > 0) {
                    nthreads = parsed;
                }
            }
            continue;
        }

        if (!arg.empty() && arg[0] != '-') {
            int parsed = 0;
            if (parse_int_strict(arg, parsed)) {
                if (npart <= 0) {
                    npart = parsed;
                } else if (t_step <= 0) {
                    t_step = parsed;
                }
            }
            continue;
        }

        if (arg.rfind("--hpx:", 0) == 0) {
            continue;
        }
    }

    if (show_help) {
        print_usage(std::cout);
        return 0;
    }

    if (parse_error || npart <= 0 || t_step <= 0) {
        print_usage(std::cerr);
        return 1;
    }

    if (nthreads <= 0) {
        nthreads = 1;
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
            if (in_space(space, particle_array[static_cast<std::size_t>(j)])) {
                octree_insert(octree, space, &particle_array[static_cast<std::size_t>(j)]);
            }
        }

        parallel_for(
            0, particle_array.size(), nthreads,
            [&](std::size_t idx) {
                particle_array[idx].force_x = 0.0;
                particle_array[idx].force_y = 0.0;
                particle_array[idx].force_z = 0.0;
            });

        parallel_for(
            0, particle_array.size(), nthreads,
            [&](std::size_t idx) {
                if (in_space(space, particle_array[idx])) {
                    compute_force(&particle_array[idx], octree);
                }
            });

        parallel_for(
            0, particle_array.size(), nthreads,
            [&](std::size_t idx) {
                if (in_space(space, particle_array[idx])) {
                    update_particle_position_and_velocity(&particle_array[idx]);
                }
            });

        free_octree(octree);

        print_summary(
            particle_array.data(), npart, space, step,
            1 /* kept for interface compatibility */);
    }

    {
        std::ofstream outfile("barnes_hut.txt", std::ios::app);
        outfile << "Num Particles: " << npart << " | Num Steps: " << t_step << "\n";
    }

    return 0;
}
