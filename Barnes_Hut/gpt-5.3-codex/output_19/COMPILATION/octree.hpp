/* ============================================================================
 * Barnes-Hut Octree (HPX version)
 * Original: https://github.com/Jrokisky/MPI-Barnes-hut, restructured + HPX
 * Author:   Justin Rokisky (original), translated to HPX execution model
 * ============================================================================ */
#pragma once

#include "particle.hpp"

#include <cmath>
#include <cstddef>

// define octree parts
struct Octree {
    Octree* children[8];
    Particle* value;
    double total_mass;
    double com_x, com_y, com_z;
    double box_size;
    int num_leaves;
    Space space;
};

Octree* create_empty_octree(Space space);

void octree_insert(Octree* octree, Space space, Particle* p);

void update_center_of_mass(Octree* octree, Particle* p);

void compute_force(Particle* leaf, Octree* octree);

void free_octree(Octree* octree);

constexpr double THETA = 1.0;
// Thanks newton.
constexpr double GRAVITY = 4.30091e-3;

inline Octree* create_empty_octree(Space space)
{
    auto* octree = new Octree();
    for (int i = 0; i < 8; i++) {
        octree->children[i] = nullptr;
    }
    octree->value = nullptr;
    octree->total_mass = 0.0;
    octree->com_x = 0.0;
    octree->com_y = 0.0;
    octree->com_z = 0.0;
    octree->box_size = space.boundary_x - space.origin_x;
    octree->num_leaves = 0;
    octree->space = space;
    return octree;
}

inline void octree_insert(Octree* octree, Space space, Particle* p)
{
    bool done = false;
    while (!done) {
        update_center_of_mass(octree, p);

        bool has_children = false;
        for (int i = 0; i < 8; i++) {
            if (octree->children[i] != nullptr) {
                has_children = true;
                break;
            }
        }

        Space sub_spaces[8];
        double mid_x = (space.origin_x + space.boundary_x) / 2.0;
        double mid_y = (space.origin_y + space.boundary_y) / 2.0;
        double mid_z = (space.origin_z + space.boundary_z) / 2.0;

        for (int i = 0; i < 8; i++) {
            bool gt_mid_x = i & (1 << 0);
            bool gt_mid_y = i & (1 << 1);
            bool gt_mid_z = i & (1 << 2);

            sub_spaces[i].boundary_x = gt_mid_x ? space.boundary_x : mid_x;
            sub_spaces[i].origin_x   = gt_mid_x ? mid_x : space.origin_x;
            sub_spaces[i].boundary_y = gt_mid_y ? space.boundary_y : mid_y;
            sub_spaces[i].origin_y   = gt_mid_y ? mid_y : space.origin_y;
            sub_spaces[i].boundary_z = gt_mid_z ? space.boundary_z : mid_z;
            sub_spaces[i].origin_z   = gt_mid_z ? mid_z : space.origin_z;
        }

        if (octree->value != nullptr) {
            Particle* curr_child = octree->value;
            octree->value = nullptr;

            int curr_child_octant = get_octant(*curr_child, space);
            if (octree->children[curr_child_octant] == nullptr) {
                Octree* new_child = create_empty_octree(sub_spaces[curr_child_octant]);
                octree->children[curr_child_octant] = new_child;
                update_center_of_mass(new_child, curr_child);
            }
            octree->children[curr_child_octant]->value = curr_child;

            int p_octant = get_octant(*p, space);

            if (p_octant != curr_child_octant) {
                if (octree->children[p_octant] == nullptr) {
                    Octree* new_child = create_empty_octree(sub_spaces[p_octant]);
                    octree->children[p_octant] = new_child;
                    update_center_of_mass(new_child, p);
                    new_child->value = p;
                } else {
                    octree_insert(octree->children[p_octant], sub_spaces[p_octant], p);
                }
                done = true;
            } else {
                space = sub_spaces[p_octant];
                octree = octree->children[p_octant];
            }
        } else if (has_children) {
            int p_octant = get_octant(*p, space);
            if (octree->children[p_octant] == nullptr) {
                octree->children[p_octant] = create_empty_octree(sub_spaces[p_octant]);
            }
            space = sub_spaces[p_octant];
            octree = octree->children[p_octant];
        } else {
            octree->value = p;
            done = true;
        }
    }
}

// Defined here because it needs full Octree definition
// (declared in particle.hpp as forward-declared)
inline void update_center_of_mass(Octree* octree, Particle* p)
{
    double total_mass = octree->total_mass + p->mass;
    octree->com_x = clamp(clamp(octree->total_mass * octree->com_x) + clamp(p->x * p->mass)) / total_mass;
    octree->com_y = clamp(clamp(octree->total_mass * octree->com_y) + clamp(p->y * p->mass)) / total_mass;
    octree->com_z = clamp(clamp(octree->total_mass * octree->com_z) + clamp(p->z * p->mass)) / total_mass;
    octree->total_mass = total_mass;
}

namespace octree_detail {

inline void compute_force_contribution(
    const Particle* leaf, const Octree* octree, double& fx, double& fy, double& fz)
{
    if (octree == nullptr) {
        return;
    }

    if (octree->value != nullptr) {
        Particle* l = octree->value;
        if (l == leaf) {
            return;
        }

        double dx = l->x - leaf->x;
        double dy = l->y - leaf->y;
        double dz = l->z - leaf->z;

        double distance = std::sqrt(
            clamp(dx * dx) +
            clamp(dy * dy) +
            clamp(dz * dz));
        distance = clamp(distance);

        double distance_cubed = clamp(distance * distance * distance);
        double scale = clamp(GRAVITY * leaf->mass * l->mass / distance_cubed);

        fx += clamp(scale * dx);
        fy += clamp(scale * dy);
        fz += clamp(scale * dz);
        return;
    }

    double dx = octree->com_x - leaf->x;
    double dy = octree->com_y - leaf->y;
    double dz = octree->com_z - leaf->z;

    double distance = std::sqrt(
        clamp(dx * dx) +
        clamp(dy * dy) +
        clamp(dz * dz));
    distance = clamp(distance);

    if ((octree->box_size / distance) < THETA) {
        double distance_cubed = clamp(distance * distance * distance);
        double scale = clamp(GRAVITY * leaf->mass * octree->total_mass / distance_cubed);

        fx += clamp(scale * dx);
        fy += clamp(scale * dy);
        fz += clamp(scale * dz);
        return;
    }

    for (int i = 0; i < 8; i++) {
        if (octree->children[i] != nullptr) {
            compute_force_contribution(leaf, octree->children[i], fx, fy, fz);
        }
    }
}

} // namespace octree_detail

inline void compute_force(Particle* leaf, Octree* octree)
{
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;

    octree_detail::compute_force_contribution(leaf, octree, fx, fy, fz);

    leaf->force_x += fx;
    leaf->force_y += fy;
    leaf->force_z += fz;
}

inline void free_octree(Octree* octree)
{
    if (octree == nullptr) {
        return;
    }

    for (int i = 0; i < 8; i++) {
        if (octree->children[i] != nullptr) {
            free_octree(octree->children[i]);
        }
    }

    delete octree;
}
