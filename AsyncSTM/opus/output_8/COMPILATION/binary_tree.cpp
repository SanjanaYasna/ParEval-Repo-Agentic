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
#include <string>
#include <vector>
#include <cstring>
#include <cassert>
#include <functional>

using namespace Legion;

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
};

enum FieldIDs {
    FID_KEY,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
};

static const int NIL_NODE = -1;
static const size_t MAX_VALUE_LEN = 64;
static const size_t MAX_NODES = 128;

// Fixed-size value type stored in the region in place of std::string
struct NodeValue {
    char data[MAX_VALUE_LEN];
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

    // Create index space, field space, and logical region for tree nodes
    Rect<1> bounds(0, MAX_NODES - 1);
    IndexSpace is = runtime->create_index_space(ctx, bounds);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int),       FID_KEY);
        fa.allocate_field(sizeof(NodeValue), FID_VALUE);
        fa.allocate_field(sizeof(int),       FID_LEFT);
        fa.allocate_field(sizeof(int),       FID_RIGHT);
    }
    LogicalRegion lr = runtime->create_logical_region(ctx, is, fs);

    // Inline-map the region with READ_WRITE / EXCLUSIVE privileges
    RegionRequirement req(lr, READ_WRITE, EXCLUSIVE, lr);
    req.add_field(FID_KEY);
    req.add_field(FID_VALUE);
    req.add_field(FID_LEFT);
    req.add_field(FID_RIGHT);
    InlineLauncher il(req);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();

    // Obtain field accessors
    const FieldAccessor<READ_WRITE, int, 1>       key_acc(pr,   FID_KEY);
    const FieldAccessor<READ_WRITE, NodeValue, 1>  value_acc(pr, FID_VALUE);
    const FieldAccessor<READ_WRITE, int, 1>       left_acc(pr,  FID_LEFT);
    const FieldAccessor<READ_WRITE, int, 1>       right_acc(pr, FID_RIGHT);

    // Tree state: root index and next-free counter
    int root_idx  = NIL_NODE;
    int num_nodes = 0;

    // ---- helper: allocate a new node in the region ----
    auto alloc_node = [&](int k, const std::string& v) -> int {
        assert(num_nodes < static_cast<int>(MAX_NODES));
        int idx = num_nodes++;
        Point<1> p(idx);
        key_acc[p] = k;
        NodeValue nv;
        memset(nv.data, 0, MAX_VALUE_LEN);
        strncpy(nv.data, v.c_str(), MAX_VALUE_LEN - 1);
        value_acc[p] = nv;
        left_acc[p]  = NIL_NODE;
        right_acc[p] = NIL_NODE;
        return idx;
    };

    // ---- helper: recursive insert ----
    std::function<void(int, const std::string&, int)> insert_at;
    insert_at = [&](int k, const std::string& v, int leaf_idx) {
        Point<1> lp(leaf_idx);
        if (k < key_acc[lp]) {
            if (left_acc[lp] != NIL_NODE)
                insert_at(k, v, left_acc[lp]);
            else
                left_acc[lp] = alloc_node(k, v);
        } else { // k >= key_acc[lp]
            if (right_acc[lp] != NIL_NODE)
                insert_at(k, v, right_acc[lp]);
            else
                right_acc[lp] = alloc_node(k, v);
        }
    };

    // ---- insert: public entry point ----
    auto insert = [&](int k, const std::string& v) {
        if (root_idx == NIL_NODE)
            root_idx = alloc_node(k, v);
        else
            insert_at(k, v, root_idx);
    };

    // ---- helper: recursive search ----
    std::function<int(int, int)> search_at;
    search_at = [&](int k, int node_idx) -> int {
        if (node_idx == NIL_NODE)
            return NIL_NODE;
        Point<1> p(node_idx);
        int node_key = key_acc[p];
        if (k == node_key)
            return node_idx;
        else if (k < node_key)
            return search_at(k, left_acc[p]);
        else
            return search_at(k, right_acc[p]);
    };

    // ---- search: public entry point ----
    auto search = [&](int k) -> int {
        return search_at(k, root_idx);
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
    for (const auto& entry : data) {
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

        if (idx != NIL_NODE) {
            Point<1> p(idx);
            NodeValue found_nv = value_acc[p];
            outfile << "Found! Result Value: \"" << found_nv.data << "\"\n";
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

    // Cleanup Legion resources
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
