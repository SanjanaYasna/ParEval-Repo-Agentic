/* ============================================================================
 * Octree data structure and Barnes-Hut force
 * ============================================================================ */
#pragma once

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
        const double mid_x = (space.origin_x + space.boundary_x) / 2.0;
        const double mid_y = (space.origin_y + space.boundary_y) / 2.0;
        const double mid_z = (space.origin_z + space.boundary_z) / 2.0;

        for (int i = 0; i < 8; i++) {
            const bool gt_mid_x = i & (1 << 0);
            const bool gt_mid_y = i & (1 << 1);
            const bool gt_mid_z = i & (1 << 2);

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
                Octree* new_child = create_empty_octree(sub_spaces[p_octant]);
                octree->children[p_octant] = new_child;
                update_center_of_mass(new_child, p);
                new_child->value = p;
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
    const double total_mass = octree->total_mass + p->mass;
    octree->com_x = clamp(clamp(octree->total_mass * octree->com_x) + clamp(p->x * p->mass)) / total_mass;
    octree->com_y = clamp(clamp(octree->total_mass * octree->com_y) + clamp(p->y * p->mass)) / total_mass;
    octree->com_z = clamp(clamp(octree->total_mass * octree->com_z) + clamp(p->z * p->mass)) / total_mass;
    octree->total_mass = total_mass;
}

namespace bh_detail {

struct ForceAccum {
    double fx{0.0};
    double fy{0.0};
    double fz{0.0};

    ForceAccum& operator+=(ForceAccum const& other)
    {
        fx += other.fx;
        fy += other.fy;
        fz += other.fz;
        return *this;
    }
};

inline ForceAccum force_from_mass(const Particle* leaf, double mass, double x, double y, double z)
{
    ForceAccum out{};
    const double distance = clamp(compute_distance(leaf, x, y, z));

    out.fx = clamp(GRAVITY * leaf->mass * mass *
                   clamp((x - leaf->x) / clamp(std::pow(distance, 3.0))));
    out.fy = clamp(GRAVITY * leaf->mass * mass *
                   clamp((y - leaf->y) / clamp(std::pow(distance, 3.0))));
    out.fz = clamp(GRAVITY * leaf->mass * mass *
                   clamp((z - leaf->z) / clamp(std::pow(distance, 3.0))));
    return out;
}

inline ForceAccum compute_force_contribution(const Particle* leaf, const Octree* octree)
{
    if (octree == nullptr) {
        return {};
    }

    if (octree->value != nullptr) {
        const Particle* other = octree->value;
        if (other == leaf) {
            return {};
        }
        return force_from_mass(leaf, other->mass, other->x, other->y, other->z);
    }

    const double distance = clamp(compute_distance(leaf, octree->com_x, octree->com_y, octree->com_z));

    if ((octree->box_size / distance) < THETA) {
        return force_from_mass(leaf, octree->total_mass, octree->com_x, octree->com_y, octree->com_z);
    }

    ForceAccum total{};
    for (int i = 0; i < 8; ++i) {
        if (octree->children[i] != nullptr) {
            total += compute_force_contribution(leaf, octree->children[i]);
        }
    }
    return total;
}

} // namespace bh_detail

inline void compute_force(Particle* leaf, Octree* octree)
{
    const bh_detail::ForceAccum f =
        bh_detail::compute_force_contribution(leaf, static_cast<const Octree*>(octree));

    leaf->force_x += clamp(f.fx);
    leaf->force_y += clamp(f.fy);
    leaf->force_z += clamp(f.fz);
}
