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
#include <cassert>
#include "legion.h"

using namespace Legion;

// Task IDs
enum {
    TOP_LEVEL_TASK_ID,
};

// Field IDs for tree node region
enum {
    FID_KEY,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
};

static const int MAX_NODES    = 128;
static const int MAX_VALUE_LEN = 64;
static const int NIL           = -1;

// POD struct for storing string values inside a Legion region field
struct NodeValue {
    char str[MAX_VALUE_LEN];
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

    // ---------------------------------------------------------------
    // Create a logical region to hold tree nodes.
    // Each entry stores: key (int), value (NodeValue), left (int), right (int)
    // Child indices are integer offsets into this same region (NIL = -1).
    // ---------------------------------------------------------------
    Rect<1> bounds(0, MAX_NODES - 1);
    IndexSpace is = runtime->create_index_space(ctx, bounds);
    runtime->attach_name(is, "tree_node_is");

    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int),       FID_KEY);
        fa.allocate_field(sizeof(NodeValue), FID_VALUE);
        fa.allocate_field(sizeof(int),       FID_LEFT);
        fa.allocate_field(sizeof(int),       FID_RIGHT);
    }
    runtime->attach_name(fs, "tree_node_fs");

    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);
    runtime->attach_name(lr, "tree_node_lr");

    // Inline-map the region for sequential read/write access
    InlineLauncher il(RegionRequirement(lr, READ_WRITE, EXCLUSIVE, lr));
    il.add_field(FID_KEY);
    il.add_field(FID_VALUE);
    il.add_field(FID_LEFT);
    il.add_field(FID_RIGHT);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();

    // Obtain field accessors
    const FieldAccessor<READ_WRITE, int,       1> acc_key  (pr, FID_KEY);
    const FieldAccessor<READ_WRITE, NodeValue, 1> acc_value(pr, FID_VALUE);
    const FieldAccessor<READ_WRITE, int,       1> acc_left (pr, FID_LEFT);
    const FieldAccessor<READ_WRITE, int,       1> acc_right(pr, FID_RIGHT);

    // Tree bookkeeping
    int next_free = 0;
    int root_idx  = NIL;

    // ----- helper: allocate a new node in the region -----
    auto alloc_node = [&](int key, const std::string &value) -> int {
        assert(next_free < MAX_NODES);
        int idx = next_free++;
        acc_key[idx] = key;
        NodeValue nv;
        memset(nv.str, 0, MAX_VALUE_LEN);
        strncpy(nv.str, value.c_str(), MAX_VALUE_LEN - 1);
        acc_value[idx] = nv;
        acc_left[idx]  = NIL;
        acc_right[idx] = NIL;
        return idx;
    };

    // ----- helper: recursive insert into subtree rooted at node_idx -----
    std::function<void(int, const std::string &, int)> insert_at;
    insert_at = [&](int key, const std::string &value, int node_idx) {
        int node_key = acc_key[node_idx];
        if (key < node_key) {
            int left = acc_left[node_idx];
            if (left != NIL)
                insert_at(key, value, left);
            else
                acc_left[node_idx] = alloc_node(key, value);
        } else {  // key >= node key  (matches original >=)
            int right = acc_right[node_idx];
            if (right != NIL)
                insert_at(key, value, right);
            else
                acc_right[node_idx] = alloc_node(key, value);
        }
    };

    // ----- public insert -----
    auto insert = [&](int key, const std::string &value) {
        if (root_idx == NIL)
            root_idx = alloc_node(key, value);
        else
            insert_at(key, value, root_idx);
    };

    // ----- helper: recursive search returning node index or NIL -----
    std::function<int(int, int)> search_at;
    search_at = [&](int key, int node_idx) -> int {
        if (node_idx == NIL) return NIL;
        int node_key = acc_key[node_idx];
        if (key == node_key) return node_idx;
        if (key < node_key)
            return search_at(key, (int)acc_left[node_idx]);
        else
            return search_at(key, (int)acc_right[node_idx]);
    };

    // ----- public search -----
    auto search = [&](int key) -> int {
        return search_at(key, root_idx);
    };

    // ===============================================================
    // 1. Data Insertion Phase
    // ===============================================================
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

    // ===============================================================
    // 2. Search Test Phase
    // ===============================================================
    outfile << "Running Search Tests:\n";

    auto perform_search = [&](int search_key) {
        outfile << "  Search(Key: " << search_key << ") -> ";
        int idx = search(search_key);
        if (idx != NIL) {
            NodeValue nv = acc_value[idx];
            outfile << "Found! Result Value: \""
                    << nv.str << "\"\n";
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

    // ---------------------------------------------------------------
    // Cleanup Legion resources
    // ---------------------------------------------------------------
    runtime->unmap_region(ctx, pr);
    runtime->destroy_logical_region(ctx, lr);
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
