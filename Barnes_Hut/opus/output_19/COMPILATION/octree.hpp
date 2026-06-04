/* ============================================================================
 * OpenMP Barnes-Hut octree
 * Original: https://github.com/Jrokisky/MPI-Barnes-hut, but restructured
 * Author:   Justin Rokisky
 * ============================================================================ */
#pragma once

#include <vector>
#include <iterator>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "particle.hpp"

// --------------------------------------------------------------------------
// Octree node
// --------------------------------------------------------------------------
struct Octree {
    Octree *children[8];
    Particle *value;
    double total_mass;
    double com_x, com_y, com_z;
    double box_size;
    int num_leaves;
    Space space;
};

// Forward declarations
Octree * create_empty_octree(Space space);
void     octree_insert(Octree *octree, Space space, Particle *p);
void     update_center_of_mass(Octree *octree, Particle *p);
void     compute_force(Particle *leaf, Octree *octree);
void     free_octree(Octree *octree);
void     compute_forces_parallel(Particle *particles, int npart,
                                 Space space, Octree *octree);

// --------------------------------------------------------------------------
// Constants
// --------------------------------------------------------------------------
constexpr double THETA   = 1.0;
constexpr double GRAVITY = 4.30091e-3;   // Newton (pc, solar-mass, km/s units)

// --------------------------------------------------------------------------
// create_empty_octree
// --------------------------------------------------------------------------
inline Octree * create_empty_octree(Space space)
{
    auto *octree = new Octree();
    for (int i = 0; i < 8; i++) {
        octree->children[i] = nullptr;
    }
    octree->value      = nullptr;
    octree->total_mass = 0.0;
    octree->com_x      = 0.0;
    octree->com_y      = 0.0;
    octree->com_z      = 0.0;
    octree->box_size   = space.boundary_x - space.origin_x;
    octree->num_leaves = 0;
    octree->space      = space;
    return octree;
}

// --------------------------------------------------------------------------
// octree_insert  (sequential – tree build is inherently serial)
// --------------------------------------------------------------------------
inline void octree_insert(Octree *octree, Space space, Particle *p)
{
    bool done = false;
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

        // Build sub-spaces
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

        if (octree->value != nullptr)   // LEAF
        {
            Particle *curr_child = octree->value;
            octree->value = nullptr;

            int curr_child_octant = get_octant(*curr_child, space);

            if (octree->children[curr_child_octant] == nullptr) {
                Octree *new_child = create_empty_octree(sub_spaces[curr_child_octant]);
                octree->children[curr_child_octant] = new_child;
                update_center_of_mass(new_child, curr_child);
            }
            octree->children[curr_child_octant]->value = curr_child;

            int p_octant = get_octant(*p, space);

            if (p_octant != curr_child_octant) {
                Octree *new_child = create_empty_octree(sub_spaces[p_octant]);
                octree->children[p_octant] = new_child;
                update_center_of_mass(new_child, p);
                new_child->value = p;
                done = true;
            }
            else {
                space  = sub_spaces[p_octant];
                octree = octree->children[p_octant];
            }
        }
        else if (has_children)           // MIDDLE LAYER
        {
            int p_octant = get_octant(*p, space);
            if (octree->children[p_octant] == nullptr) {
                octree->children[p_octant] = create_empty_octree(sub_spaces[p_octant]);
            }
            space  = sub_spaces[p_octant];
            octree = octree->children[p_octant];
        }
        else {                           // EMPTY LEAF (Root)
            octree->value = p;
            done = true;
        }
    }
}

// --------------------------------------------------------------------------
// free_octree
// --------------------------------------------------------------------------
inline void free_octree(Octree *octree)
{
    if (octree == nullptr) return;
    for (int i = 0; i < 8; i++) {
        if (octree->children[i] != nullptr) {
            free_octree(octree->children[i]);
        }
    }
    delete octree;
}

// --------------------------------------------------------------------------
// update_center_of_mass
// --------------------------------------------------------------------------
inline void update_center_of_mass(Octree *octree, Particle *p)
{
    double total_mass = octree->total_mass + p->mass;
    octree->com_x = clamp(clamp(octree->total_mass * octree->com_x) +
                          clamp(p->x * p->mass)) / total_mass;
    octree->com_y = clamp(clamp(octree->total_mass * octree->com_y) +
                          clamp(p->y * p->mass)) / total_mass;
    octree->com_z = clamp(clamp(octree->total_mass * octree->com_z) +
                          clamp(p->z * p->mass)) / total_mass;
    octree->total_mass = total_mass;
}

// --------------------------------------------------------------------------
// compute_force  (single particle against the tree – read-only on tree)
// --------------------------------------------------------------------------
inline void compute_force(Particle *leaf, Octree *octree)
{
    if (octree->value != nullptr) {
        Particle *l = octree->value;
        if (l == leaf) return;   // no self-interaction

        double distance = clamp(compute_distance(leaf, l->x, l->y, l->z));

        leaf->force_x += clamp(GRAVITY * leaf->mass * l->mass *
                               clamp((l->x - leaf->x) / clamp(pow(distance, 3.0))));
        leaf->force_y += clamp(GRAVITY * leaf->mass * l->mass *
                               clamp((l->y - leaf->y) / clamp(pow(distance, 3.0))));
        leaf->force_z += clamp(GRAVITY * leaf->mass * l->mass *
                               clamp((l->z - leaf->z) / clamp(pow(distance, 3.0))));
    }
    else {
        double distance = clamp(compute_distance(leaf, octree->com_x,
                                                 octree->com_y, octree->com_z));

        if ((octree->box_size / distance) < THETA) {
            leaf->force_x += clamp(GRAVITY * leaf->mass * octree->total_mass *
                                   clamp((octree->com_x - leaf->x) / clamp(pow(distance, 3.0))));
            leaf->force_y += clamp(GRAVITY * leaf->mass * octree->total_mass *
                                   clamp((octree->com_y - leaf->y) / clamp(pow(distance, 3.0))));
            leaf->force_z += clamp(GRAVITY * leaf->mass * octree->total_mass *
                                   clamp((octree->com_z - leaf->z) / clamp(pow(distance, 3.0))));
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

// --------------------------------------------------------------------------
// compute_forces_parallel
//   Reset forces, compute forces, and update positions/velocities for ALL
//   particles using OpenMP parallelism.
//   The octree is read-only at this point, so concurrent traversal is safe.
// --------------------------------------------------------------------------
inline void compute_forces_parallel(Particle *particles, int npart,
                                    Space space, Octree *octree)
{
    #pragma omp parallel for schedule(dynamic, 100)
    for (int idx = 0; idx < npart; idx++) {
        Particle *p = &particles[idx];
        if (!in_space(space, *p)) continue;

        // Phase 0: reset forces
        p->force_x = 0.0;
        p->force_y = 0.0;
        p->force_z = 0.0;

        // Phase 1: compute forces (read-only tree traversal)
        compute_force(p, octree);

        // Phase 2: leapfrog integration
        update_particle_position_and_velocity(p);
    }
}
