/* ============================================================================
 * Original: https://github.com/Jrokisky/MPI-Barnes-hut, but restructured
 * Author:   Justin Rokisky
 * Translated to C++17 threads execution model.
 * ============================================================================ */
#pragma once

#include "particle.hpp"

// Define octree parts
struct Octree {
    Octree *children[8];
    Particle *value;
    double total_mass;
    double com_x, com_y, com_z;
    double box_size;
    int num_leaves;
    Space space;
};

Octree * create_empty_octree(Space space);

void octree_insert(Octree *octree, Space space, Particle *p);

void update_center_of_mass(Octree *octree, Particle *p);

void compute_force(Particle *leaf, Octree *octree);

void free_octree(Octree *octree);

constexpr double THETA   = 1.0;
// Thanks newton.
constexpr double GRAVITY = 4.30091e-3;

inline Octree * create_empty_octree(Space space)
{
    auto *octree = new Octree();
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

inline void octree_insert(Octree *octree, Space space, Particle *p){
    bool done = false;
    // Originally used recursion, but trying to resolve performance issues.
    while (!done)
    {

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
    double mid_x = (space.origin_x + space.boundary_x)/2.0;

    double mid_y = (space.origin_y + space.boundary_y)/2.0;
    double mid_z = (space.origin_z + space.boundary_z)/2.0;

    // BOTTOM    TOP
    //  __ __   __ __
    // |2 |3 | |6 |7 |
    // |__|__| |__|__|
    // |0 |1 | |4 |5 |
    // |__|__| |__|__|

    for (int i = 0; i < 8; i++) {
        // Check if this octant is greater than the midpoint on different axis.
        bool gt_mid_x = i & (1 << 0);
        bool gt_mid_y = i & (1 << 1);
        bool gt_mid_z = i & (1 << 2);

        sub_spaces[i].boundary_x = gt_mid_x ? space.boundary_x : mid_x;
        sub_spaces[i].origin_x = gt_mid_x ? mid_x : space.origin_x;
        sub_spaces[i].boundary_y = gt_mid_y ? space.boundary_y : mid_y;
        sub_spaces[i].origin_y = gt_mid_y ? mid_y : space.origin_y;
        sub_spaces[i].boundary_z = gt_mid_z ? space.boundary_z : mid_z;
        sub_spaces[i].origin_z = gt_mid_z ? mid_z : space.origin_z;
    }

    if (octree->value != nullptr) // LEAF.
    {
        // Stash the current particle and then clear it.
        Particle *curr_child = octree->value;
        octree->value = nullptr;
        // Handle current child particle.
        int curr_child_octant = get_octant(*curr_child, space);

        if (octree->children[curr_child_octant] == nullptr) {
            Octree * new_child  = create_empty_octree(sub_spaces[curr_child_octant]);
            octree->children[curr_child_octant] = new_child;
            update_center_of_mass(new_child, curr_child);
        }
        octree->children[curr_child_octant]->value = curr_child;

        // Insert new particle.
        int p_octant = get_octant(*p, space);

        // Old and new particle end up in different branches.
        if (p_octant != curr_child_octant) {
            Octree * new_child  = create_empty_octree(sub_spaces[p_octant]);
            octree->children[p_octant] = new_child;
            update_center_of_mass(new_child, p);
            new_child->value = p;
            done = true;
        }
        else {
            // Same branch, so repeat.
            space = sub_spaces[p_octant];
            octree = octree->children[p_octant];
        }
    }
    else if(has_children) // MIDDLE LAYER.
    {
        // Insert new particle.
        int p_octant = get_octant(*p, space);
        if (octree->children[p_octant] == nullptr) {
            octree->children[p_octant] = create_empty_octree(sub_spaces[p_octant]);
        }
        space = sub_spaces[p_octant];
        octree = octree->children[p_octant];
    }
    else { // EMPTY LEAF (Root)
        octree->value = p;
        done = true;
    }

    }
}

inline void free_octree(Octree *octree)
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
inline void update_center_of_mass(Octree *octree, Particle *p)
{
    double total_mass = octree->total_mass + p->mass;
    octree->com_x = clamp( clamp(octree->total_mass * octree->com_x) + clamp(p->x * p->mass)) / total_mass;
    octree->com_y = clamp( clamp(octree->total_mass * octree->com_y) + clamp(p->y * p->mass)) / total_mass;
    octree->com_z = clamp( clamp(octree->total_mass * octree->com_z) + clamp(p->z * p->mass)) / total_mass;
    // remember to write back the mass back into the octree
    octree->total_mass = total_mass;
}

// compute_force is read-only on the octree structure, so it is safe to call
// concurrently from multiple threads for different leaf particles.
inline void compute_force(Particle *leaf, Octree *octree)
{
    // When we get to a leaf.
    if (octree->value != nullptr) {
        Particle *l = octree->value;
        if (l == leaf) return; // no self-interaction
        double distance = clamp(compute_distance(leaf, l->x, l->y, l->z));

        leaf->force_x += clamp(GRAVITY * leaf->mass * l->mass * clamp((l->x - leaf->x) / clamp(pow(distance, 3.0))));
        leaf->force_y += clamp(GRAVITY * leaf->mass * l->mass * clamp((l->y - leaf->y) / clamp(pow(distance, 3.0))));
        leaf->force_z += clamp(GRAVITY * leaf->mass * l->mass * clamp((l->z - leaf->z) / clamp(pow(distance, 3.0))));
    }
    else {
        double distance = clamp(compute_distance(leaf, octree->com_x, octree->com_y, octree->com_z));

        // Use center of mass of octant.
        if ((octree->box_size / distance) < THETA) {
            leaf->force_x += clamp(GRAVITY * leaf->mass * octree->total_mass * clamp((octree->com_x - leaf->x) / clamp(pow(distance, 3.0))));
            leaf->force_y += clamp(GRAVITY * leaf->mass * octree->total_mass * clamp((octree->com_y - leaf->y) / clamp(pow(distance, 3.0))));
            leaf->force_z += clamp(GRAVITY * leaf->mass * octree->total_mass * clamp((octree->com_z - leaf->z) / clamp(pow(distance, 3.0))));
        }
        else {
            for (int i = 0; i < 8; i++) {
                if (octree->children[i] != nullptr) {
                    compute_force(leaf, octree->children[i]);
                }
            }
        }
    }
}
