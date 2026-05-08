////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <functional>
#include "legion.h"

using namespace Legion;

static const int NULL_IDX = -1;
static const int MAX_NODES = 128;
static const int MAX_VALUE_LEN = 64;

enum FieldIDs {
    FID_KEY,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
};

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
};

// Fixed-size string suitable for storage in a Legion logical region field.
struct FixedString {
    char data[MAX_VALUE_LEN];

    FixedString() { memset(data, 0, MAX_VALUE_LEN); }
    FixedString(const std::string& s) {
        memset(data, 0, MAX_VALUE_LEN);
        strncpy(data, s.c_str(), MAX_VALUE_LEN - 1);
    }
};

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

    // --- Create logical region to hold tree nodes ---
    Rect<1> bounds(0, MAX_NODES - 1);
    IndexSpace is = runtime->create_index_space(ctx, bounds);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int),         FID_KEY);
        fa.allocate_field(sizeof(FixedString), FID_VALUE);
        fa.allocate_field(sizeof(int),         FID_LEFT);
        fa.allocate_field(sizeof(int),         FID_RIGHT);
    }
    LogicalRegion tree_lr = runtime->create_logical_region(ctx, is, fs);

    // Inline-map the region for direct read/write access
    RegionRequirement req(tree_lr, READ_WRITE, EXCLUSIVE, tree_lr);
    req.add_field(FID_KEY);
    req.add_field(FID_VALUE);
    req.add_field(FID_LEFT);
    req.add_field(FID_RIGHT);
    InlineLauncher il(req);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();

    // Obtain field accessors
    const FieldAccessor<READ_WRITE, int, 1>         acc_key(pr, FID_KEY);
    const FieldAccessor<READ_WRITE, FixedString, 1> acc_val(pr, FID_VALUE);
    const FieldAccessor<READ_WRITE, int, 1>         acc_left(pr, FID_LEFT);
    const FieldAccessor<READ_WRITE, int, 1>         acc_right(pr, FID_RIGHT);

    // Initialize every slot
    for (int i = 0; i < MAX_NODES; i++) {
        acc_key[i]   = 0;
        acc_val[i]   = FixedString();
        acc_left[i]  = NULL_IDX;
        acc_right[i] = NULL_IDX;
    }

    // Tree bookkeeping (replaces shared_var<node*> root)
    int root_idx  = NULL_IDX;
    int next_free = 0;

    // ---- Tree helper lambdas (mirror the original struct methods) ----

    // Allocate a new node in the region, return its index
    auto alloc_node = [&](int key, const std::string& value) -> int {
        int idx = next_free++;
        acc_key[idx]   = key;
        acc_val[idx]   = FixedString(value);
        acc_left[idx]  = NULL_IDX;
        acc_right[idx] = NULL_IDX;
        return idx;
    };

    // Recursive insert (mirrors binary_tree::insert(key, value, leaf, t))
    std::function<void(int, const std::string&, int)> insert_at;
    insert_at = [&](int key, const std::string& value, int node_idx) {
        int node_key = acc_key[node_idx];
        if (key < node_key) {
            int left = acc_left[node_idx];
            if (left != NULL_IDX)
                insert_at(key, value, left);
            else
                acc_left[node_idx] = alloc_node(key, value);
        } else {  // key >= node_key  (matches original >=)
            int right = acc_right[node_idx];
            if (right != NULL_IDX)
                insert_at(key, value, right);
            else
                acc_right[node_idx] = alloc_node(key, value);
        }
    };

    // Public insert (mirrors binary_tree::insert)
    auto insert = [&](int key, const std::string& value) {
        if (root_idx == NULL_IDX)
            root_idx = alloc_node(key, value);
        else
            insert_at(key, value, root_idx);
    };

    // Recursive search – returns node index or NULL_IDX
    std::function<int(int, int)> search_at;
    search_at = [&](int key, int node_idx) -> int {
        if (node_idx == NULL_IDX)
            return NULL_IDX;
        int node_key = acc_key[node_idx];
        if (key == node_key)
            return node_idx;
        if (key < node_key)
            return search_at(key, static_cast<int>(acc_left[node_idx]));
        return search_at(key, static_cast<int>(acc_right[node_idx]));
    };

    // Public search (mirrors binary_tree::search)
    auto search = [&](int key) -> int {
        return search_at(key, root_idx);
    };

    // ---- Main logic (identical to the original) ----

    {
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
            insert(entry.id, entry.val);
        }
        outfile << "\n";

        // Search Test Phase
        outfile << "Running Search Tests:\n";

        auto perform_search = [&](int search_key) {
            outfile << "  Search(Key: " << search_key << ") -> ";
            int idx = search(search_key);
            if (idx != NULL_IDX) {
                FixedString val = acc_val[idx];
                outfile << "Found! Result Value: \"" << val.data << "\"\n";
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
    }

    outfile.close();

    // Cleanup Legion resources
    runtime->unmap_region(ctx, pr);
    runtime->destroy_logical_region(ctx, tree_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}

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
