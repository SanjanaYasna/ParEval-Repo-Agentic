/* ============================================================================
 * HPX-adapted octree.hpp
 * Original: https://github.com/Jrokisky/MPI-Barnes-hut, restructured
 * Author:   Justin Rokisky
 * ============================================================================ */
#pragma once

#include <array>
#include <cmath>

#include "particle.hpp"

// Define octree parts
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

constexpr double THETA   = 1.0;
// Thanks Newton.
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
    // Iterative insert to avoid recursive overhead.
    while (!done)
    {
        update_center_of_mass(octree, p);

        bool has_children = false;
        for (int i = 0; i < 8; i++) {
            if (octree->children[i] != nullptr) {
                has_children = true;
                break;
            }
        }

        std::array<Space, 8> sub_spaces{};
        double mid_x = (space.origin_x + space.boundary_x) / 2.0;
        double mid_y = (space.origin_y + space.boundary_y) / 2.0;
        double mid_z = (space.origin_z + space.boundary_z) / 2.0;

        // BOTTOM    TOP
        //  __ __   __ __
        // |2 |3 | |6 |7 |
        // |__|__| |__|__|
        // |0 |1 | |4 |5 |
        // |__|__| |__|__|
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

        if (octree->value != nullptr) // LEAF
        {
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
                Octree* new_child = create_empty_octree(sub_spaces[p_octant]);
                octree->children[p_octant] = new_child;
                update_center_of_mass(new_child, p);
                new_child->value = p;
                done = true;
            } else {
                space = sub_spaces[p_octant];
                octree = octree->children[p_octant];
            }
        }
        else if (has_children) // INTERNAL NODE
        {
            int p_octant = get_octant(*p, space);
            if (octree->children[p_octant] == nullptr) {
                octree->children[p_octant] = create_empty_octree(sub_spaces[p_octant]);
            }
            space = sub_spaces[p_octant];
            octree = octree->children[p_octant];
        }
        else // EMPTY LEAF
        {
            octree->value = p;
            done = true;
        }
    }
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

namespace detail {

struct ForceAccum {
    double fx{0.0};
    double fy{0.0};
    double fz{0.0};

    ForceAccum& operator+=(ForceAccum const& rhs)
    {
        fx += rhs.fx;
        fy += rhs.fy;
        fz += rhs.fz;
        return *this;
    }
};

inline double distance_xyz(double x1, double y1, double z1,
                           double x2, double y2, double z2)
{
    return clamp(std::sqrt(
        clamp((x1 - x2) * (x1 - x2)) +
        clamp((y1 - y2) * (y1 - y2)) +
        clamp((z1 - z2) * (z1 - z2))
    ));
}

inline ForceAccum compute_force_contribution(Particle const* leaf, Octree const* octree)
{
    ForceAccum accum{};
    if (octree == nullptr) {
        return accum;
    }

    // Leaf node interaction
    if (octree->value != nullptr) {
        Particle const* other = octree->value;
        if (other == leaf) {
            return accum; // no self-interaction
        }

        double distance = distance_xyz(leaf->x, leaf->y, leaf->z, other->x, other->y, other->z);
        double denom = clamp(std::pow(distance, 3.0));

        accum.fx = clamp(GRAVITY * leaf->mass * other->mass * clamp((other->x - leaf->x) / denom));
        accum.fy = clamp(GRAVITY * leaf->mass * other->mass * clamp((other->y - leaf->y) / denom));
        accum.fz = clamp(GRAVITY * leaf->mass * other->mass * clamp((other->z - leaf->z) / denom));
        return accum;
    }

    if (octree->total_mass <= 0.0) {
        return accum;
    }

    // Internal node approximation
    double distance = distance_xyz(leaf->x, leaf->y, leaf->z,
                                   octree->com_x, octree->com_y, octree->com_z);

    if ((octree->box_size / distance) < THETA) {
        double denom = clamp(std::pow(distance, 3.0));
        accum.fx = clamp(GRAVITY * leaf->mass * octree->total_mass * clamp((octree->com_x - leaf->x) / denom));
        accum.fy = clamp(GRAVITY * leaf->mass * octree->total_mass * clamp((octree->com_y - leaf->y) / denom));
        accum.fz = clamp(GRAVITY * leaf->mass * octree->total_mass * clamp((octree->com_z - leaf->z) / denom));
        return accum;
    }

    for (int i = 0; i < 8; ++i) {
        if (octree->children[i] != nullptr) {
            accum += compute_force_contribution(leaf, octree->children[i]);
        }
    }

    return accum;
}

} // namespace detail

inline void compute_force(Particle* leaf, Octree* octree)
{
    detail::ForceAccum acc = detail::compute_force_contribution(leaf, octree);
    leaf->force_x += acc.fx;
    leaf->force_y += acc.fy;
    leaf->force_z += acc.fz;
}
