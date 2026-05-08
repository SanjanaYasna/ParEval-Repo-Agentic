////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Translated from HPX/ASTM to Legion execution model.
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <functional>
#include "legion.h"

using namespace Legion;

// Field IDs for tree node data stored in a logical region
enum FieldIDs {
    FID_KEY = 0,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
};

// Task IDs
enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
};

static const int MAX_VALUE_LEN = 64;
static const int MAX_NODES = 128;
static const int NULL_IDX = -1;

// Fixed-size value type suitable for storage in a Legion region field
struct NodeValue {
    char data[MAX_VALUE_LEN];
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
    // -------------------------------------------------------------------------
    // Create a logical region to hold the binary tree nodes.
    // Each entry in the region represents a tree node with key, value, and
    // child indices (analogous to the original shared_var<node*> pointers).
    // -------------------------------------------------------------------------
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, MAX_NODES - 1));
    runtime->attach_name(is, "tree_node_index_space");

    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_KEY);
        fa.allocate_field(sizeof(NodeValue), FID_VALUE);
        fa.allocate_field(sizeof(int), FID_LEFT);
        fa.allocate_field(sizeof(int), FID_RIGHT);
    }
    runtime->attach_name(fs, "tree_node_field_space");

    LogicalRegion tree_lr = runtime->create_logical_region(ctx, is, fs);
    runtime->attach_name(tree_lr, "tree_region");

    // Inline-map the region with READ_WRITE/EXCLUSIVE privilege.
    // In Legion, exclusive access within a task provides the same atomicity
    // guarantees as the ASTM transactions in the original code.
    RegionRequirement req(tree_lr, READ_WRITE, EXCLUSIVE, tree_lr);
    req.add_field(FID_KEY);
    req.add_field(FID_VALUE);
    req.add_field(FID_LEFT);
    req.add_field(FID_RIGHT);
    InlineLauncher il(req);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();

    const FieldAccessor<READ_WRITE, int, 1> key_acc(pr, FID_KEY);
    const FieldAccessor<READ_WRITE, NodeValue, 1> value_acc(pr, FID_VALUE);
    const FieldAccessor<READ_WRITE, int, 1> left_acc(pr, FID_LEFT);
    const FieldAccessor<READ_WRITE, int, 1> right_acc(pr, FID_RIGHT);

    // Initialize all node slots
    for (int i = 0; i < MAX_NODES; i++) {
        key_acc[i] = 0;
        NodeValue empty_nv;
        memset(empty_nv.data, 0, MAX_VALUE_LEN);
        value_acc[i] = empty_nv;
        left_acc[i] = NULL_IDX;
        right_acc[i] = NULL_IDX;
    }

    int root_idx = NULL_IDX;
    int next_node = 0;

    // -------------------------------------------------------------------------
    // Allocate a new node in the region (analogous to "new node(key, value)")
    // -------------------------------------------------------------------------
    auto alloc_node = [&](int k, const std::string &v) -> int {
        int idx = next_node++;
        key_acc[idx] = k;
        NodeValue nv;
        memset(nv.data, 0, MAX_VALUE_LEN);
        strncpy(nv.data, v.c_str(), MAX_VALUE_LEN - 1);
        value_acc[idx] = nv;
        left_acc[idx] = NULL_IDX;
        right_acc[idx] = NULL_IDX;
        return idx;
    };

    // -------------------------------------------------------------------------
    // Insert: traverses the tree stored in the region and inserts a new node.
    // The ASTM transaction retry loop is unnecessary in Legion because the
    // task holds exclusive access to the region, preventing conflicts.
    // -------------------------------------------------------------------------
    std::function<void(int, const std::string &, int)> insert_at;
    insert_at = [&](int k, const std::string &v, int node_idx) {
        if (k < key_acc[node_idx]) {
            if (left_acc[node_idx] != NULL_IDX)
                insert_at(k, v, left_acc[node_idx]);
            else
                left_acc[node_idx] = alloc_node(k, v);
        } else { // key >= current node's key (matches original >=)
            if (right_acc[node_idx] != NULL_IDX)
                insert_at(k, v, right_acc[node_idx]);
            else
                right_acc[node_idx] = alloc_node(k, v);
        }
    };

    auto insert = [&](int k, const std::string &v) {
        if (root_idx == NULL_IDX)
            root_idx = alloc_node(k, v);
        else
            insert_at(k, v, root_idx);
    };

    // -------------------------------------------------------------------------
    // Search: traverses the tree in the region looking for a key.
    // Returns the node index, or NULL_IDX if not found.
    // -------------------------------------------------------------------------
    std::function<int(int, int)> search_at;
    search_at = [&](int k, int node_idx) -> int {
        if (node_idx == NULL_IDX)
            return NULL_IDX;
        if (k == key_acc[node_idx])
            return node_idx;
        if (k < key_acc[node_idx])
            return search_at(k, left_acc[node_idx]);
        else
            return search_at(k, right_acc[node_idx]);
    };

    auto search = [&](int k) -> int {
        return search_at(k, root_idx);
    };

    // -------------------------------------------------------------------------
    // Main program logic: mirrors the original binary_tree.cpp main()
    // -------------------------------------------------------------------------
    std::ofstream outfile("binary_tree.txt");
    if (!outfile.is_open()) {
        std::cerr << "Failed to open binary_tree.txt for writing." << std::endl;
        // Cleanup before returning
        runtime->unmap_region(ctx, pr);
        runtime->destroy_logical_region(ctx, tree_lr);
        runtime->destroy_field_space(ctx, fs);
        runtime->destroy_index_space(ctx, is);
        return;
    }

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
    for (const auto &entry : data) {
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
            NodeValue nv = value_acc[idx];
            outfile << "Found! Result Value: \"" << nv.data << "\"\n";
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

    outfile.close();

    // -------------------------------------------------------------------------
    // Cleanup Legion resources
    // -------------------------------------------------------------------------
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
