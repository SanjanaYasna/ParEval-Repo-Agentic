/* ============================================================================
 * Barnes-Hut Octree
 * ============================================================================ */
#pragma once

#include <cmath>

#include "particle.hpp"

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
            bool gt_mid_x = (i & (1 << 0)) != 0;
            bool gt_mid_y = (i & (1 << 1)) != 0;
            bool gt_mid_z = (i & (1 << 2)) != 0;

            sub_spaces[i].boundary_x = gt_mid_x ? space.boundary_x : mid_x;
            sub_spaces[i].origin_x = gt_mid_x ? mid_x : space.origin_x;
            sub_spaces[i].boundary_y = gt_mid_y ? space.boundary_y : mid_y;
            sub_spaces[i].origin_y = gt_mid_y ? mid_y : space.origin_y;
            sub_spaces[i].boundary_z = gt_mid_z ? space.boundary_z : mid_z;
            sub_spaces[i].origin_z = gt_mid_z ? mid_z : space.origin_z;
        }

        if (octree->value != nullptr) {
            Particle* curr_child = octree->value;
            octree->value = nullptr;

            int curr_child_octant = get_octant(*curr_child, space);
            if (octree->children[curr_child_octant] == nullptr) {
                Octree* new_child = create_empty_octree(sub_spaces[curr_child_octant]);
                octree->children[curr_child_octant] = new_child;
                update_center_of_mass(new_child, curr_child);
                new_child->value = curr_child;
            } else {
                octree_insert(octree->children[curr_child_octant], sub_spaces[curr_child_octant], curr_child);
            }

            int p_octant = get_octant(*p, space);

            if (p_octant != curr_child_octant) {
                if (octree->children[p_octant] == nullptr) {
                    Octree* new_child = create_empty_octree(sub_spaces[p_octant]);
                    octree->children[p_octant] = new_child;
                }
                update_center_of_mass(octree->children[p_octant], p);
                octree->children[p_octant]->value = p;
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

inline void update_center_of_mass(Octree* octree, Particle* p)
{
    double total_mass = octree->total_mass + p->mass;
    octree->com_x = clamp(clamp(octree->total_mass * octree->com_x) + clamp(p->x * p->mass)) / total_mass;
    octree->com_y = clamp(clamp(octree->total_mass * octree->com_y) + clamp(p->y * p->mass)) / total_mass;
    octree->com_z = clamp(clamp(octree->total_mass * octree->com_z) + clamp(p->z * p->mass)) / total_mass;
    octree->total_mass = total_mass;
}

namespace bh_hpx_detail {

struct ForceContribution {
    double fx{0.0};
    double fy{0.0};
    double fz{0.0};

    ForceContribution& operator+=(const ForceContribution& other)
    {
        fx += other.fx;
        fy += other.fy;
        fz += other.fz;
        return *this;
    }
};

inline bool particle_inside_space(const Particle* p, const Space& s)
{
    return (p->x > s.origin_x) && (p->x < s.boundary_x) &&
           (p->y > s.origin_y) && (p->y < s.boundary_y) &&
           (p->z > s.origin_z) && (p->z < s.boundary_z);
}

inline ForceContribution compute_force_contribution(Particle* leaf, Octree* octree)
{
    ForceContribution out{};

    if (octree == nullptr || octree->total_mass <= 0.0) {
        return out;
    }

    if (octree->value != nullptr) {
        Particle* l = octree->value;
        if (l == leaf) {
            return out;
        }

        double distance = clamp(compute_distance(leaf, l->x, l->y, l->z));
        double distance3 = clamp(distance * distance * distance);

        out.fx = clamp(GRAVITY * leaf->mass * l->mass * clamp((l->x - leaf->x) / distance3));
        out.fy = clamp(GRAVITY * leaf->mass * l->mass * clamp((l->y - leaf->y) / distance3));
        out.fz = clamp(GRAVITY * leaf->mass * l->mass * clamp((l->z - leaf->z) / distance3));
        return out;
    }

    bool leaf_in_node = particle_inside_space(leaf, octree->space);

    double distance = clamp(compute_distance(leaf, octree->com_x, octree->com_y, octree->com_z));
    if (!leaf_in_node && (octree->box_size / distance) < THETA) {
        double distance3 = clamp(distance * distance * distance);
        out.fx = clamp(GRAVITY * leaf->mass * octree->total_mass * clamp((octree->com_x - leaf->x) / distance3));
        out.fy = clamp(GRAVITY * leaf->mass * octree->total_mass * clamp((octree->com_y - leaf->y) / distance3));
        out.fz = clamp(GRAVITY * leaf->mass * octree->total_mass * clamp((octree->com_z - leaf->z) / distance3));
        return out;
    }

    for (int i = 0; i < 8; i++) {
        Octree* child = octree->children[i];
        if (child != nullptr) {
            out += compute_force_contribution(leaf, child);
        }
    }

    return out;
}

} // namespace bh_hpx_detail

inline void compute_force(Particle* leaf, Octree* octree)
{
    bh_hpx_detail::ForceContribution fc = bh_hpx_detail::compute_force_contribution(leaf, octree);
    leaf->force_x += fc.fx;
    leaf->force_y += fc.fy;
    leaf->force_z += fc.fz;
}
