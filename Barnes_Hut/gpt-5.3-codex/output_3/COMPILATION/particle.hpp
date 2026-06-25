/* ============================================================================
 * Particle utilities for Barnes-Hut
 * ============================================================================ */
#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <random>
#include <thread>
#include <type_traits>
#include <vector>

struct Particle {
    int id;
    double x, y, z;
    double vel_x, vel_y, vel_z;
    double force_x, force_y, force_z;
    double mass;

    template <typename Archive>
    void serialize(Archive& ar, unsigned int /*version*/)
    {
        ar& id& x& y& z& vel_x& vel_y& vel_z& force_x& force_y& force_z& mass;
    }
};

struct Space {
    double boundary_x, boundary_y, boundary_z;
    double origin_x, origin_y, origin_z;

    template <typename Archive>
    void serialize(Archive& ar, unsigned int /*version*/)
    {
        ar& boundary_x& boundary_y& boundary_z& origin_x& origin_y& origin_z;
    }
};

// forward declarations from octree.hpp
struct Octree;

double drand_custom(double low, double high);
double clamp(double in);

// Get the octant that the given particle belongs in.
int get_octant(Particle particle, Space space);

// Generate random particles in the given space.
void generate_random_particles(Particle* p, Space s, int count);

// Print net particle info:
void print_summary(Particle* particles, int npart, Space space, int timestep, int nprocs);

// utils
double compute_distance(Particle* particle, double com_x, double com_y, double com_z);
void update_particle_position_and_velocity(Particle* p);
bool in_space(Space s, Particle p);

void update_center_of_mass(Octree* octree, Particle* p);
void compute_force(Particle* leaf, Octree* octree);

inline unsigned int& num_threads_storage()
{
    static unsigned int threads = []() {
        unsigned int n = std::thread::hardware_concurrency();
        return n == 0 ? 1u : n;
    }();
    return threads;
}

inline void set_num_threads(unsigned int n)
{
    if (n > 0) {
        num_threads_storage() = n;
    }
}

inline unsigned int get_num_threads()
{
    return num_threads_storage();
}

template <typename Func>
inline void parallel_for(int begin, int end, Func&& fn)
{
    int total = end - begin;
    if (total <= 0) {
        return;
    }

    unsigned int nthreads = get_num_threads();
    if (nthreads <= 1 || total < 1024) {
        for (int i = begin; i < end; ++i) {
            fn(i);
        }
        return;
    }

    nthreads = std::min<unsigned int>(nthreads, static_cast<unsigned int>(total));
    if (nthreads <= 1) {
        for (int i = begin; i < end; ++i) {
            fn(i);
        }
        return;
    }

    using FnType = typename std::decay<Func>::type;
    FnType func = std::forward<Func>(fn);

    std::vector<std::thread> workers;
    workers.reserve(nthreads);

    int base = total / static_cast<int>(nthreads);
    int rem = total % static_cast<int>(nthreads);
    int start = begin;

    for (unsigned int t = 0; t < nthreads; ++t) {
        int chunk = base + (static_cast<int>(t) < rem ? 1 : 0);
        int s = start;
        int e = s + chunk;
        start = e;

        workers.emplace_back([s, e, &func]() {
            for (int i = s; i < e; ++i) {
                func(i);
            }
        });
    }

    for (auto& th : workers) {
        th.join();
    }
}

inline double drand_custom(double low, double high)
{
    double range = high - low;
    return (static_cast<double>(rand()) * range) / static_cast<double>(RAND_MAX) + low;
}

inline double clamp(double in)
{
    if (in >= 0.0) {
        double tmp = in < DBL_MIN ? DBL_MIN : in;
        return tmp > DBL_MAX ? DBL_MAX : tmp;
    } else {
        double tmp = in < -DBL_MAX ? -DBL_MAX : in;
        return tmp > -DBL_MIN ? -DBL_MIN : tmp;
    }
}

inline int get_octant(Particle particle, Space space)
{
    int octant = 0;
    if (particle.x > ((space.origin_x + space.boundary_x) / 2.0)) {
        octant |= (1 << 0);
    }
    if (particle.y > ((space.origin_y + space.boundary_y) / 2.0)) {
        octant |= (1 << 1);
    }
    if (particle.z > ((space.origin_z + space.boundary_z) / 2.0)) {
        octant |= (1 << 2);
    }
    return octant;
}

inline void generate_random_particles(Particle* p, Space /*space*/, int count)
{
    constexpr std::uint64_t base_seed = 42ull;

    parallel_for(0, count, [p](int i) {
        std::mt19937_64 gen(base_seed + 0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(i));
        std::uniform_real_distribution<double> dist(0.0, 100.0);

        Particle& part = p[i];
        part.x = dist(gen);
        part.y = dist(gen);
        part.z = dist(gen);
        part.vel_x = 0.0;
        part.vel_y = 0.0;
        part.vel_z = 0.0;
        part.force_x = 0.0;
        part.force_y = 0.0;
        part.force_z = 0.0;
        part.mass = 1000000.0;
        part.id = i;
    });
}

// print statistic summaries as a net among particles at a given timestep
inline void print_summary(Particle* particles, int npart, Space space, int timestep, int nprocs)
{
    (void)nprocs; // retained for API compatibility

    double total_ke = 0.0;
    double total_px = 0.0;
    double total_py = 0.0;
    double total_pz = 0.0;
    double com_x_acc = 0.0;
    double com_y_acc = 0.0;
    double com_z_acc = 0.0;
    double total_mass = 0.0;
    int active = 0;

    for (int i = 0; i < npart; ++i) {
        const Particle& p = particles[i];
        if (!in_space(space, p)) {
            continue;
        }

        ++active;
        double v2 = p.vel_x * p.vel_x + p.vel_y * p.vel_y + p.vel_z * p.vel_z;
        total_ke += 0.5 * p.mass * v2;
        total_px += p.mass * p.vel_x;
        total_py += p.mass * p.vel_y;
        total_pz += p.mass * p.vel_z;
        com_x_acc += p.mass * p.x;
        com_y_acc += p.mass * p.y;
        com_z_acc += p.mass * p.z;
        total_mass += p.mass;
    }

    double com_x = 0.0, com_y = 0.0, com_z = 0.0;
    if (total_mass > 0.0) {
        com_x = com_x_acc / total_mass;
        com_y = com_y_acc / total_mass;
        com_z = com_z_acc / total_mass;
    }

    double total_p = std::sqrt(total_px * total_px + total_py * total_py + total_pz * total_pz);

    std::ofstream outfile("barnes_hut.txt", std::ios::app);
    outfile << "Step " << timestep << " | active " << active << "/" << npart
            << " | KE " << std::scientific << std::setprecision(2) << total_ke << " | |p| "
            << std::scientific << std::setprecision(2) << total_p << " | CoM (" << std::fixed
            << std::setprecision(2) << com_x << ", " << com_y << ", " << com_z << ")\n";
}

inline double compute_distance(Particle* particle, double com_x, double com_y, double com_z)
{
    return std::sqrt(clamp(std::pow(particle->x - com_x, 2)) +
                     clamp(std::pow(particle->y - com_y, 2)) +
                     clamp(std::pow(particle->z - com_z, 2)));
}

inline void update_particle_position_and_velocity(Particle* p)
{
    double dt = 0.001;
    double acc_x = clamp(p->force_x / p->mass);
    double acc_y = clamp(p->force_y / p->mass);
    double acc_z = clamp(p->force_z / p->mass);

    // Compute middle of timestep velocity.
    double v_mid_x = p->vel_x + clamp(0.5 * dt * acc_x);
    double v_mid_y = p->vel_y + clamp(0.5 * dt * acc_y);
    double v_mid_z = p->vel_z + clamp(0.5 * dt * acc_z);

    // Compute new position.
    p->x = p->x + clamp(dt * v_mid_x);
    p->y = p->y + clamp(dt * v_mid_y);
    p->z = p->z + clamp(dt * v_mid_z);

    // Compute second half of velocity.
    p->vel_x = v_mid_x + clamp(0.5 * dt * acc_x);
    p->vel_y = v_mid_y + clamp(0.5 * dt * acc_y);
    p->vel_z = v_mid_z + clamp(0.5 * dt * acc_z);
}

inline bool in_space(Space s, Particle p)
{
    return (p.x > s.origin_x) && (p.x < s.boundary_x) && (p.y > s.origin_y) &&
           (p.y < s.boundary_y) && (p.z > s.origin_z) && (p.z < s.boundary_z);
}
