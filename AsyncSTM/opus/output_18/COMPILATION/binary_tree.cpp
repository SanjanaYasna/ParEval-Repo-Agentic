////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include "legion.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <functional>
#include <cassert>

using namespace Legion;

// Constants
static const int MAX_NODES = 128;
static const int MAX_VALUE_LEN = 64;
static const int NULL_IDX = -1;

// Fixed-size value type for storing strings in a Legion region field
struct NodeValue {
    char data[MAX_VALUE_LEN];
};

// Field IDs for tree node fields stored in the logical region
enum FieldIDs {
    FID_KEY,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
};

// Task IDs
enum TaskIDs {
    TOP_LEVEL_TASK_ID,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // Open output file
    std::ofstream outfile("binary_tree.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
        return;
    }

    // Create a logical region to store tree nodes.
    // Each node occupies one entry in the region, indexed by an integer.
    // Node pointers (left/right children) are represented as integer indices
    // into this region, with NULL_IDX (-1) representing a null pointer.
    // This replaces the original shared_var<node*> pattern from the STM model.
    Rect<1> node_rect(0, MAX_NODES - 1);
    IndexSpace is = runtime->create_index_space(ctx, node_rect);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_KEY);
        fa.allocate_field(sizeof(NodeValue), FID_VALUE);
        fa.allocate_field(sizeof(int), FID_LEFT);
        fa.allocate_field(sizeof(int), FID_RIGHT);
    }
    LogicalRegion tree_lr = runtime->create_logical_region(ctx, is, fs);

    // Inline map the region for read/write access.
    // In the original code, each insert/search was wrapped in a transaction
    // (do { ... } while (!t.commit_transaction())).  In Legion, the runtime
    // guarantees coherent, exclusive access through the region requirement,
    // so no retry loop is needed.
    RegionRequirement req(tree_lr, READ_WRITE, EXCLUSIVE, tree_lr);
    req.add_field(FID_KEY);
    req.add_field(FID_VALUE);
    req.add_field(FID_LEFT);
    req.add_field(FID_RIGHT);
    InlineLauncher il(req);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();

    const FieldAccessor<READ_WRITE, int, 1> acc_key(pr, FID_KEY);
    const FieldAccessor<READ_WRITE, NodeValue, 1> acc_value(pr, FID_VALUE);
    const FieldAccessor<READ_WRITE, int, 1> acc_left(pr, FID_LEFT);
    const FieldAccessor<READ_WRITE, int, 1> acc_right(pr, FID_RIGHT);

    // Initialize all node slots
    for (int i = 0; i < MAX_NODES; i++) {
        acc_key[i] = 0;
        NodeValue empty_val;
        memset(empty_val.data, 0, MAX_VALUE_LEN);
        acc_value[i] = empty_val;
        acc_left[i] = NULL_IDX;
        acc_right[i] = NULL_IDX;
    }

    // Tree state: replaces the shared_var<node*> root member of binary_tree
    int next_free = 0;
    int root_idx = NULL_IDX;

    // Allocate a new node in the region (replaces `new node(key, value)`)
    auto alloc_node = [&](int key, const std::string& value) -> int {
        assert(next_free < MAX_NODES);
        int idx = next_free++;
        acc_key[idx] = key;
        NodeValue nv;
        memset(nv.data, 0, MAX_VALUE_LEN);
        strncpy(nv.data, value.c_str(), MAX_VALUE_LEN - 1);
        acc_value[idx] = nv;
        acc_left[idx] = NULL_IDX;
        acc_right[idx] = NULL_IDX;
        return idx;
    };

    // Recursive insert helper
    // Replaces the private insert(Key, Value, shared_var<node*>&, transaction&)
    std::function<void(int, const std::string&, int)> insert_at;
    insert_at = [&](int key, const std::string& value, int leaf_idx) {
        if (key < acc_key[leaf_idx]) {
            int left = acc_left[leaf_idx];
            if (left != NULL_IDX)
                insert_at(key, value, left);
            else
                acc_left[leaf_idx] = alloc_node(key, value);
        } else { // key >= leaf key (matches original >= branch)
            int right = acc_right[leaf_idx];
            if (right != NULL_IDX)
                insert_at(key, value, right);
            else
                acc_right[leaf_idx] = alloc_node(key, value);
        }
    };

    // Insert a key-value pair into the tree
    // Replaces binary_tree::insert which wrapped work in a transaction
    auto insert = [&](int key, const std::string& value) {
        if (root_idx == NULL_IDX) {
            root_idx = alloc_node(key, value);
        } else {
            insert_at(key, value, root_idx);
        }
    };

    // Recursive search helper
    // Replaces the private search(Key, shared_var<node*>&, transaction&)
    std::function<int(int, int)> search_at;
    search_at = [&](int key, int node_idx) -> int {
        if (node_idx == NULL_IDX) return NULL_IDX;
        int node_key = acc_key[node_idx];
        if (key == node_key) return node_idx;
        if (key < node_key)
            return search_at(key, acc_left[node_idx]);
        else
            return search_at(key, acc_right[node_idx]);
    };

    // Search for a key in the tree, returns node index or NULL_IDX
    // Replaces binary_tree::search which wrapped work in a transaction
    auto search = [&](int key) -> int {
        return search_at(key, root_idx);
    };

    // ===================================================================
    // Main logic (matches original main() body)
    // ===================================================================
    {
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
            insert(entry.id, entry.val);
        }
        outfile << "\n";

        // 2. Search Test Phase
        outfile << "Running Search Tests:\n";

        auto perform_search = [&](int search_key) {
            outfile << "  Search(Key: " << search_key << ") -> ";

            int idx = search(search_key);

            if (idx != NULL_IDX) {
                NodeValue nv = acc_value[idx];
                outfile << "Found! Result Value: \""
                        << nv.data << "\"\n";
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
