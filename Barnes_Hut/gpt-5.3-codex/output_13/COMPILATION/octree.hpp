/* ============================================================================
 * octree.hpp
 * Original: https://github.com/Jrokisky/MPI-Barnes-hut, restructured
 * ============================================================================ */
#pragma once

#include "particle.hpp"

#include <array>

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

constexpr double THETA   = 1.0;
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
    // Iterative insertion.
    while (!done) {
        update_center_of_mass(octree, p);

        // Check if we have any children.
        bool has_children = false;
        for (int i = 0; i < 8; i++) {
            if (octree->children[i] != nullptr) {
                has_children = true;
                break;
            }
        }

        // Build subspaces
        Space sub_spaces[8];
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

        if (octree->value != nullptr) // LEAF.
        {
            // Stash the current particle and then clear it.
            Particle* curr_child = octree->value;
            octree->value = nullptr;

            // Handle current child particle.
            int curr_child_octant = get_octant(*curr_child, space);

            if (octree->children[curr_child_octant] == nullptr) {
                Octree* new_child = create_empty_octree(sub_spaces[curr_child_octant]);
                octree->children[curr_child_octant] = new_child;
                update_center_of_mass(new_child, curr_child);
            }
            octree->children[curr_child_octant]->value = curr_child;

            // Insert new particle.
            int p_octant = get_octant(*p, space);

            // Old and new particle end up in different branches.
            if (p_octant != curr_child_octant) {
                Octree* new_child = create_empty_octree(sub_spaces[p_octant]);
                octree->children[p_octant] = new_child;
                update_center_of_mass(new_child, p);
                new_child->value = p;
                done = true;
            } else {
                // Same branch, so repeat.
                space = sub_spaces[p_octant];
                octree = octree->children[p_octant];
            }
        } else if (has_children) // MIDDLE LAYER.
        {
            int p_octant = get_octant(*p, space);
            if (octree->children[p_octant] == nullptr) {
                octree->children[p_octant] = create_empty_octree(sub_spaces[p_octant]);
            }
            space = sub_spaces[p_octant];
            octree = octree->children[p_octant];
        } else { // EMPTY LEAF (Root)
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

namespace detail {

inline std::array<double, 3> force_from_mass_point(
    Particle* leaf, double mass, double x, double y, double z)
{
    std::array<double, 3> f{0.0, 0.0, 0.0};

    double distance = clamp(compute_distance(leaf, x, y, z));
    double dist3 = clamp(std::pow(distance, 3.0));

    f[0] = clamp(GRAVITY * leaf->mass * mass * clamp((x - leaf->x) / dist3));
    f[1] = clamp(GRAVITY * leaf->mass * mass * clamp((y - leaf->y) / dist3));
    f[2] = clamp(GRAVITY * leaf->mass * mass * clamp((z - leaf->z) / dist3));
    return f;
}

// Recursive force traversal returning contributions.
inline std::array<double, 3> compute_force_contrib(Particle* leaf, Octree* octree)
{
    if (octree == nullptr) {
        return {0.0, 0.0, 0.0};
    }

    // Reached a leaf in the tree.
    if (octree->value != nullptr) {
        Particle* l = octree->value;
        if (l == leaf) {
            return {0.0, 0.0, 0.0}; // no self-interaction
        }
        return force_from_mass_point(leaf, l->mass, l->x, l->y, l->z);
    }

    // Internal node: decide BH approximation vs descend.
    double distance = clamp(compute_distance(leaf, octree->com_x, octree->com_y, octree->com_z));
    if ((octree->box_size / distance) < THETA) {
        return force_from_mass_point(leaf, octree->total_mass, octree->com_x, octree->com_y, octree->com_z);
    }

    // Descend into children.
    std::array<double, 3> acc{0.0, 0.0, 0.0};
    for (int i = 0; i < 8; i++) {
        if (octree->children[i] != nullptr) {
            auto tmp = compute_force_contrib(leaf, octree->children[i]);
            acc[0] += tmp[0];
            acc[1] += tmp[1];
            acc[2] += tmp[2];
        }
    }

    return acc;
}

} // namespace detail

inline void compute_force(Particle* leaf, Octree* octree)
{
    auto f = detail::compute_force_contrib(leaf, octree);
    leaf->force_x += f[0];
    leaf->force_y += f[1];
    leaf->force_z += f[2];
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
