/* ============================================================================
 * Barnes-Hut octree
 * ============================================================================ */
#pragma once

#include <cmath>
#include <cstddef>

#include "particle.hpp"

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
constexpr double GRAVITY = 4.30091e-3;
constexpr int MAX_INSERT_DEPTH = 128;

namespace detail {

struct force_accumulator {
    double fx{0.0};
    double fy{0.0};
    double fz{0.0};

    force_accumulator& operator+=(force_accumulator const& other)
    {
        fx = clamp(fx + other.fx);
        fy = clamp(fy + other.fy);
        fz = clamp(fz + other.fz);
        return *this;
    }
};

inline force_accumulator compute_force_impl(Particle const* leaf, Octree const* octree)
{
    force_accumulator out{};

    if (octree == nullptr) {
        return out;
    }

    if (octree->value != nullptr) {
        Particle const* l = octree->value;
        if (l == leaf) {
            return out;
        }

        double distance = clamp(compute_distance(leaf, l->x, l->y, l->z));
        double inv_d3 = clamp(1.0 / clamp(distance * distance * distance));
        double coeff = clamp(GRAVITY * leaf->mass * l->mass);

        out.fx = clamp(coeff * clamp((l->x - leaf->x) * inv_d3));
        out.fy = clamp(coeff * clamp((l->y - leaf->y) * inv_d3));
        out.fz = clamp(coeff * clamp((l->z - leaf->z) * inv_d3));
        return out;
    }

    double distance = clamp(compute_distance(leaf, octree->com_x, octree->com_y, octree->com_z));

    if ((octree->box_size / distance) < THETA) {
        double inv_d3 = clamp(1.0 / clamp(distance * distance * distance));
        double coeff = clamp(GRAVITY * leaf->mass * octree->total_mass);

        out.fx = clamp(coeff * clamp((octree->com_x - leaf->x) * inv_d3));
        out.fy = clamp(coeff * clamp((octree->com_y - leaf->y) * inv_d3));
        out.fz = clamp(coeff * clamp((octree->com_z - leaf->z) * inv_d3));
        return out;
    }

    for (int i = 0; i < 8; ++i) {
        if (octree->children[i] != nullptr) {
            out += compute_force_impl(leaf, octree->children[i]);
        }
    }

    return out;
}

inline void free_octree_impl(Octree* octree)
{
    if (octree == nullptr) {
        return;
    }

    for (int i = 0; i < 8; ++i) {
        if (octree->children[i] != nullptr) {
            free_octree_impl(octree->children[i]);
        }
    }

    delete octree;
}

} // namespace detail

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
    int depth = 0;

    while (!done) {
        if (depth++ > MAX_INSERT_DEPTH) {
            p->x = std::nextafter(p->x, space.boundary_x);
            p->y = std::nextafter(p->y, space.boundary_y);
            p->z = std::nextafter(p->z, space.boundary_z);
            depth = 0;
        }

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
            if (curr_child == p) {
                return;
            }

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
                }
                update_center_of_mass(octree->children[p_octant], p);
                octree->children[p_octant]->value = p;
                done = true;
            } else {
                if (curr_child->x == p->x && curr_child->y == p->y && curr_child->z == p->z) {
                    p->x = std::nextafter(p->x, space.boundary_x);
                    p->y = std::nextafter(p->y, space.boundary_y);
                    p->z = std::nextafter(p->z, space.boundary_z);

                    p_octant = get_octant(*p, space);
                    if (p_octant != curr_child_octant) {
                        if (octree->children[p_octant] == nullptr) {
                            Octree* new_child = create_empty_octree(sub_spaces[p_octant]);
                            octree->children[p_octant] = new_child;
                        }
                        update_center_of_mass(octree->children[p_octant], p);
                        octree->children[p_octant]->value = p;
                        done = true;
                        continue;
                    }
                }

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
    detail::free_octree_impl(octree);
}

inline void update_center_of_mass(Octree* octree, Particle* p)
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

inline void compute_force(Particle* leaf, Octree* octree)
{
    detail::force_accumulator acc = detail::compute_force_impl(leaf, octree);
    leaf->force_x = clamp(leaf->force_x + acc.fx);
    leaf->force_y = clamp(leaf->force_y + acc.fy);
    leaf->force_z = clamp(leaf->force_z + acc.fz);
}
