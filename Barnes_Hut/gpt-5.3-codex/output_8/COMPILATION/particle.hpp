/* ============================================================================
 * Original: https://github.com/Jrokisky/MPI-Barnes-hut, but restructured
 * Author:   Justin Rokisky
 * ============================================================================ */
#pragma once

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>

struct Particle {
    int id;
    double x, y, z;
    double vel_x, vel_y, vel_z;
    double force_x, force_y, force_z;
    double mass;

    template <typename Archive>
    void serialize(Archive& ar, unsigned /*version*/)
    {
        ar& id& x& y& z& vel_x& vel_y& vel_z& force_x& force_y& force_z& mass;
    }
};

struct Space {
    double boundary_x, boundary_y, boundary_z;
    double origin_x, origin_y, origin_z;

    template <typename Archive>
    void serialize(Archive& ar, unsigned /*version*/)
    {
        ar& boundary_x& boundary_y& boundary_z& origin_x& origin_y& origin_z;
    }
};

struct Octree;

double drand_custom(double low, double high);
double clamp(double in);

int get_octant(Particle particle, Space space);
void generate_random_particles(Particle* p, Space s, int count);

void print_summary(Particle* particles, int npart, Space space,
                   int timestep, int nprocs);

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
    for (int i = 0; i < count; i++) {
        double x = drand_custom(0.0, 100.0);
        double y = drand_custom(0.0, 100.0);
        double z = drand_custom(0.0, 100.0);

        p[i].x = x;
        p[i].y = y;
        p[i].z = z;
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

inline void print_summary(Particle* particles, int npart, Space space,
                          int timestep, int nprocs)
{
    (void)nprocs;

    std::ofstream outfile("barnes_hut.txt", std::ios::app);
    double total_ke = 0.0;
    double total_px = 0.0;
    double total_py = 0.0;
    double total_pz = 0.0;
    double com_x = 0.0;
    double com_y = 0.0;
    double com_z = 0.0;
    double total_mass = 0.0;
    int active = 0;

    for (int i = 0; i < npart; i++) {
        Particle p = particles[i];
        if (!in_space(space, p)) continue;
        active++;

        double v2 = p.vel_x * p.vel_x + p.vel_y * p.vel_y + p.vel_z * p.vel_z;

        total_ke += 0.5 * p.mass * v2;
        total_px += p.mass * p.vel_x;
        total_py += p.mass * p.vel_y;
        total_pz += p.mass * p.vel_z;
        com_x += p.mass * p.x;
        com_y += p.mass * p.y;
        com_z += p.mass * p.z;
        total_mass += p.mass;
    }

    if (total_mass > 0.0) {
        com_x /= total_mass;
        com_y /= total_mass;
        com_z /= total_mass;
    }

    double total_p = std::sqrt(total_px * total_px + total_py * total_py + total_pz * total_pz);

    outfile << "Step " << timestep
            << " | active " << active << "/" << npart
            << " | KE " << std::scientific << std::setprecision(2) << total_ke
            << " | |p| " << std::scientific << std::setprecision(2) << total_p
            << " | CoM (" << std::fixed << std::setprecision(2)
            << com_x << ", " << com_y << ", " << com_z << ")\n";
}

inline double compute_distance(Particle* particle, double com_x, double com_y, double com_z)
{
    return std::sqrt(clamp(std::pow(particle->x - com_x, 2))
                   + clamp(std::pow(particle->y - com_y, 2))
                   + clamp(std::pow(particle->z - com_z, 2)));
}

inline void update_particle_position_and_velocity(Particle* p)
{
    double dt = 0.001;
    double acc_x = clamp(p->force_x / p->mass);
    double acc_y = clamp(p->force_y / p->mass);
    double acc_z = clamp(p->force_z / p->mass);

    double v_mid_x = p->vel_x + clamp(0.5 * dt * acc_x);
    double v_mid_y = p->vel_y + clamp(0.5 * dt * acc_y);
    double v_mid_z = p->vel_z + clamp(0.5 * dt * acc_z);

    p->x = p->x + clamp(dt * v_mid_x);
    p->y = p->y + clamp(dt * v_mid_y);
    p->z = p->z + clamp(dt * v_mid_z);

    p->vel_x = v_mid_x + clamp(0.5 * dt * acc_x);
    p->vel_y = v_mid_y + clamp(0.5 * dt * acc_y);
    p->vel_z = v_mid_z + clamp(0.5 * dt * acc_z);
}

inline bool in_space(Space s, Particle p)
{
    return (p.x > s.origin_x)
        && (p.x < s.boundary_x)
        && (p.y > s.origin_y)
        && (p.y < s.boundary_y)
        && (p.z > s.origin_z)
        && (p.z < s.boundary_z);
}
