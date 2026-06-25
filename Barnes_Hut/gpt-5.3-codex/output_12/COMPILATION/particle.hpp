/* ============================================================================
 * Barnes-Hut particle/domain utilities (HPX version)
 * Original MPI-based code by Justin Rokisky, adapted for HPX execution model.
 * ============================================================================ */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>

struct Particle {
    int id;
    double x, y, z;
    double vel_x, vel_y, vel_z;
    double force_x, force_y, force_z;
    double mass;
};

struct Space {
    double boundary_x, boundary_y, boundary_z;
    double origin_x, origin_y, origin_z;
};

// forward declarations from octree.hpp
struct Octree;

double drand_custom(double low, double high);
double clamp(double in);

// Get the octant that the given particle belongs in.
int get_octant(Particle particle, Space space);

// Generate random particles in the given space.
void generate_random_particles(Particle* p, Space s, int count);

// Print net particle info.
void print_summary(Particle* particles, int npart, Space space,
                   int timestep, int nlocalities);

// utils
double compute_distance(Particle* particle, double com_x, double com_y, double com_z);

void update_particle_position_and_velocity(Particle* p);

bool in_space(Space s, Particle p);

void update_center_of_mass(Octree* octree, Particle* p);
void compute_force(Particle* leaf, Octree* octree);

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
    if (particle.x > ((space.origin_x + space.boundary_x) / 2.0)) octant |= (1 << 0);
    if (particle.y > ((space.origin_y + space.boundary_y) / 2.0)) octant |= (1 << 1);
    if (particle.z > ((space.origin_z + space.boundary_z) / 2.0)) octant |= (1 << 2);
    return octant;
}

// Deterministic per-index RNG helpers.
inline std::uint64_t splitmix64(std::uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline double u01_from_u64(std::uint64_t x)
{
    // 53-bit mantissa uniform in [0, 1)
    return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0);
}

inline double drand_indexed(std::uint64_t seed, double low, double high)
{
    return low + (high - low) * u01_from_u64(splitmix64(seed));
}

inline void generate_random_particles(Particle* p, Space s, int count)
{
    (void) s;
    constexpr std::uint64_t base_seed = 42ull;

    for (int i = 0; i < count; ++i) {
        const std::uint64_t si = static_cast<std::uint64_t>(i);

        p[i].x = drand_indexed(base_seed ^ (si * 3ull + 0ull), 0.0, 100.0);
        p[i].y = drand_indexed(base_seed ^ (si * 3ull + 1ull), 0.0, 100.0);
        p[i].z = drand_indexed(base_seed ^ (si * 3ull + 2ull), 0.0, 100.0);

        p[i].vel_x = 0.0;
        p[i].vel_y = 0.0;
        p[i].vel_z = 0.0;

        p[i].force_x = 0.0;
        p[i].force_y = 0.0;
        p[i].force_z = 0.0;

        p[i].mass = 1000000.0;
        p[i].id = i;
    }
}

struct SummaryStats
{
    double total_ke = 0.0;
    double total_px = 0.0;
    double total_py = 0.0;
    double total_pz = 0.0;
    double com_x = 0.0;  // stores sum(m*x) until normalized
    double com_y = 0.0;  // stores sum(m*y) until normalized
    double com_z = 0.0;  // stores sum(m*z) until normalized
    double total_mass = 0.0;
    std::uint64_t active = 0;
};

inline void print_summary(Particle* particles, int npart, Space space,
                          int timestep, int nlocalities)
{
    (void) nlocalities;

    SummaryStats global_stats{};

    for (int i = 0; i < npart; ++i) {
        const Particle& p = particles[i];
        if (!in_space(space, p)) {
            continue;
        }

        global_stats.active += 1;
        const double v2 = p.vel_x * p.vel_x + p.vel_y * p.vel_y + p.vel_z * p.vel_z;
        global_stats.total_ke += 0.5 * p.mass * v2;
        global_stats.total_px += p.mass * p.vel_x;
        global_stats.total_py += p.mass * p.vel_y;
        global_stats.total_pz += p.mass * p.vel_z;
        global_stats.com_x += p.mass * p.x;
        global_stats.com_y += p.mass * p.y;
        global_stats.com_z += p.mass * p.z;
        global_stats.total_mass += p.mass;
    }

    double com_x = 0.0, com_y = 0.0, com_z = 0.0;
    if (global_stats.total_mass > 0.0) {
        com_x = global_stats.com_x / global_stats.total_mass;
        com_y = global_stats.com_y / global_stats.total_mass;
        com_z = global_stats.com_z / global_stats.total_mass;
    }

    const double total_p = std::sqrt(global_stats.total_px * global_stats.total_px +
                                     global_stats.total_py * global_stats.total_py +
                                     global_stats.total_pz * global_stats.total_pz);

    std::ofstream outfile("barnes_hut.txt", std::ios::app);
    outfile << "Step " << timestep
            << " | active " << global_stats.active << "/" << npart
            << " | KE " << std::scientific << std::setprecision(2) << global_stats.total_ke
            << " | |p| " << std::scientific << std::setprecision(2) << total_p
            << " | CoM (" << std::fixed << std::setprecision(2)
            << com_x << ", " << com_y << ", " << com_z << ")\n";
    outfile.close();
}

inline double compute_distance(Particle* particle, double com_x, double com_y, double com_z)
{
    return std::sqrt(clamp(std::pow(particle->x - com_x, 2)) +
                     clamp(std::pow(particle->y - com_y, 2)) +
                     clamp(std::pow(particle->z - com_z, 2)));
}

inline void update_particle_position_and_velocity(Particle* p)
{
    constexpr double dt = 0.001;
    const double acc_x = clamp(p->force_x / p->mass);
    const double acc_y = clamp(p->force_y / p->mass);
    const double acc_z = clamp(p->force_z / p->mass);

    // Compute middle of timestep velocity.
    const double v_mid_x = p->vel_x + clamp(0.5 * dt * acc_x);
    const double v_mid_y = p->vel_y + clamp(0.5 * dt * acc_y);
    const double v_mid_z = p->vel_z + clamp(0.5 * dt * acc_z);

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
    return (p.x > s.origin_x) && (p.x < s.boundary_x) &&
           (p.y > s.origin_y) && (p.y < s.boundary_y) &&
           (p.z > s.origin_z) && (p.z < s.boundary_z);
}
