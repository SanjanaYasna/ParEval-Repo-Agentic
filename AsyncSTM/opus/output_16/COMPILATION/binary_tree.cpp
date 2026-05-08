////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>

#include "legion.h"

using namespace Legion;

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
};

enum FieldIDs {
    FID_KEY,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
    FID_VALID,
};

static const int MAX_NODES = 128;
static const int MAX_VALUE_LEN = 64;

// Fixed-size string type suitable for storage in a Legion region field
struct FixedString {
    char data[MAX_VALUE_LEN];

    FixedString() { memset(data, 0, sizeof(data)); }
    FixedString(const std::string& s) {
        memset(data, 0, sizeof(data));
        strncpy(data, s.c_str(), MAX_VALUE_LEN - 1);
    }
};

// Binary tree operations implemented over Legion region accessors.
// Nodes are stored in a flat array within the region; child links are
// represented as integer indices (-1 denotes a null link).
template <typename KeyAcc, typename ValAcc, typename BoolAcc>
class BinaryTreeOps {
public:
    static void init_node(const KeyAcc& keys, const ValAcc& values,
                          const KeyAcc& lefts, const KeyAcc& rights,
                          const BoolAcc& valids,
                          int idx, int key, const std::string& value)
    {
        keys[idx]   = key;
        values[idx] = FixedString(value);
        lefts[idx]  = -1;
        rights[idx] = -1;
        valids[idx] = true;
    }

    static void insert(const KeyAcc& keys, const ValAcc& values,
                       const KeyAcc& lefts, const KeyAcc& rights,
                       const BoolAcc& valids,
                       int& node_count, int key, const std::string& value)
    {
        if (node_count == 0) {
            init_node(keys, values, lefts, rights, valids, 0, key, value);
            node_count = 1;
        } else {
            insert_at(keys, values, lefts, rights, valids,
                      node_count, 0, key, value);
        }
    }

    // Returns the index of the node whose key matches, or -1 if not found.
    static int search(const KeyAcc& keys, const KeyAcc& lefts,
                      const KeyAcc& rights, int node_idx, int key)
    {
        if (node_idx == -1)
            return -1;

        int nk = keys[node_idx];
        if (key == nk) return node_idx;
        if (key < nk)  return search(keys, lefts, rights, lefts[node_idx], key);
        return search(keys, lefts, rights, rights[node_idx], key);
    }

private:
    static void insert_at(const KeyAcc& keys, const ValAcc& values,
                          const KeyAcc& lefts, const KeyAcc& rights,
                          const BoolAcc& valids,
                          int& node_count, int node_idx,
                          int key, const std::string& value)
    {
        if (key < keys[node_idx]) {
            int left = lefts[node_idx];
            if (left != -1) {
                insert_at(keys, values, lefts, rights, valids,
                          node_count, left, key, value);
            } else {
                int new_idx = node_count++;
                init_node(keys, values, lefts, rights, valids,
                          new_idx, key, value);
                lefts[node_idx] = new_idx;
            }
        } else {
            // key >= keys[node_idx] — matches original >= branch
            int right = rights[node_idx];
            if (right != -1) {
                insert_at(keys, values, lefts, rights, valids,
                          node_count, right, key, value);
            } else {
                int new_idx = node_count++;
                init_node(keys, values, lefts, rights, valids,
                          new_idx, key, value);
                rights[node_idx] = new_idx;
            }
        }
    }
};

// -----------------------------------------------------------------------
// Top-level Legion task — mirrors the original main()
// -----------------------------------------------------------------------
void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // Open the output file
    std::ofstream outfile("binary_tree.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
        return;
    }

    {
        // --- Create the region that backs the tree node array ---------------
        Rect<1> bounds(0, MAX_NODES - 1);
        IndexSpace is = runtime->create_index_space(ctx, bounds);
        runtime->attach_name(is, "tree_nodes_is");

        FieldSpace fs = runtime->create_field_space(ctx);
        runtime->attach_name(fs, "tree_nodes_fs");
        {
            FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
            fa.allocate_field(sizeof(int),         FID_KEY);
            fa.allocate_field(sizeof(FixedString), FID_VALUE);
            fa.allocate_field(sizeof(int),         FID_LEFT);
            fa.allocate_field(sizeof(int),         FID_RIGHT);
            fa.allocate_field(sizeof(bool),        FID_VALID);
        }

        LogicalRegion tree_lr = runtime->create_logical_region(ctx, is, fs);
        runtime->attach_name(tree_lr, "tree_region");

        // Inline-map the entire region with READ_WRITE privilege
        InlineLauncher il(RegionRequirement(tree_lr, READ_WRITE,
                                            EXCLUSIVE, tree_lr));
        il.add_field(FID_KEY);
        il.add_field(FID_VALUE);
        il.add_field(FID_LEFT);
        il.add_field(FID_RIGHT);
        il.add_field(FID_VALID);
        PhysicalRegion pr = runtime->map_region(ctx, il);
        pr.wait_until_valid();

        // Obtain typed accessors
        const FieldAccessor<READ_WRITE, int, 1, coord_t,
            Realm::AffineAccessor<int, 1, coord_t>>         fa_key(pr, FID_KEY);
        const FieldAccessor<READ_WRITE, FixedString, 1, coord_t,
            Realm::AffineAccessor<FixedString, 1, coord_t>> fa_value(pr, FID_VALUE);
        const FieldAccessor<READ_WRITE, int, 1, coord_t,
            Realm::AffineAccessor<int, 1, coord_t>>         fa_left(pr, FID_LEFT);
        const FieldAccessor<READ_WRITE, int, 1, coord_t,
            Realm::AffineAccessor<int, 1, coord_t>>         fa_right(pr, FID_RIGHT);
        const FieldAccessor<READ_WRITE, bool, 1, coord_t,
            Realm::AffineAccessor<bool, 1, coord_t>>        fa_valid(pr, FID_VALID);

        // Convenience type aliases matching the accessor types above
        using KeyAcc  = decltype(fa_key);
        using ValAcc  = decltype(fa_value);
        using BoolAcc = decltype(fa_valid);

        using TreeOps = BinaryTreeOps<KeyAcc, ValAcc, BoolAcc>;

        // Initialise every slot to "empty"
        for (int i = 0; i < MAX_NODES; i++) {
            fa_valid[i] = false;
            fa_left[i]  = -1;
            fa_right[i] = -1;
        }

        int node_count = 0;

        // 1. Data Insertion Phase
        struct DataEntry {
            int id;
            std::string val;
        };

        std::vector<DataEntry> data = {
            {50, "Root Node"},
            {30, "Left Child"},
            {70, "Right Child"},
            {20, "Left-Left Grandchild"},
            {40, "Left-Right Grandchild"},
            {60, "Right-Left Grandchild"},
            {80, "Right-Right Grandchild"}
        };

        outfile << "Inserting Data:\n";
        for (const auto& entry : data) {
            outfile << "  Insert(Key: " << entry.id
                    << ", Value: \"" << entry.val << "\")" << std::endl;
            TreeOps::insert(fa_key, fa_value, fa_left, fa_right, fa_valid,
                            node_count, entry.id, entry.val);
        }
        outfile << "\n";

        // 2. Search Test Phase
        outfile << "Running Search Tests:\n";

        auto perform_search = [&](int search_key) {
            outfile << "  Search(Key: " << search_key << ") -> ";

            int idx = -1;
            if (node_count > 0)
                idx = TreeOps::search(fa_key, fa_left, fa_right,
                                      0, search_key);

            if (idx != -1) {
                outfile << "Found! Result Value: \""
                        << fa_value[idx].data << "\"\n";
            } else {
                outfile << "Not Found.\n";
            }
        };

        perform_search(99);
        perform_search(10);
        perform_search(50);
        perform_search(20);
        perform_search(60);
        perform_search(80);
        perform_search(45);

        // --- Clean up Legion resources --------------------------------------
        runtime->unmap_region(ctx, pr);
        runtime->destroy_logical_region(ctx, tree_lr);
        runtime->destroy_field_space(ctx, fs);
        runtime->destroy_index_space(ctx, is);
    }

    outfile.close();
}

// -----------------------------------------------------------------------
// main — register the top-level task and start the Legion runtime
// -----------------------------------------------------------------------
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
    }
    return Runtime::start(argc, argv);
}
