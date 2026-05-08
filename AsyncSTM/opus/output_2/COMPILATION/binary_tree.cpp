////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt 
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cstring>
#include <cassert>
#include <vector>
#include <string>
#include <functional>

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
};

static const int NULL_IDX = -1;
static const int MAX_NODES = 128;
static const int MAX_VALUE_LEN = 64;

// Fixed-size value type suitable for storage in a Legion region
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

    // Create index space and field space for tree nodes
    IndexSpace is = runtime->create_index_space(ctx, Rect<1>(0, MAX_NODES - 1));
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int), FID_KEY);
        fa.allocate_field(sizeof(NodeValue), FID_VALUE);
        fa.allocate_field(sizeof(int), FID_LEFT);
        fa.allocate_field(sizeof(int), FID_RIGHT);
    }
    LogicalRegion tree_lr = runtime->create_logical_region(ctx, is, fs);

    // Inline map the region for direct read/write access
    // This is analogous to the transactional access in the ASTM version,
    // providing exclusive access to the tree data within this task.
    InlineLauncher il(RegionRequirement(tree_lr, READ_WRITE, EXCLUSIVE, tree_lr));
    il.add_field(FID_KEY);
    il.add_field(FID_VALUE);
    il.add_field(FID_LEFT);
    il.add_field(FID_RIGHT);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();

    const FieldAccessor<READ_WRITE, int, 1> fa_key(pr, FID_KEY);
    const FieldAccessor<READ_WRITE, NodeValue, 1> fa_val(pr, FID_VALUE);
    const FieldAccessor<READ_WRITE, int, 1> fa_left(pr, FID_LEFT);
    const FieldAccessor<READ_WRITE, int, 1> fa_right(pr, FID_RIGHT);

    // Tree metadata (analogous to shared_var<node*> root in the original)
    int root_idx = NULL_IDX;
    int next_free = 0;

    // Helper: allocate a new node in the region
    // Replaces `new node(key, value)` from the ASTM version
    auto alloc_node = [&](int k, const std::string &v) -> int {
        int idx = next_free++;
        assert(idx < MAX_NODES);
        fa_key[idx] = k;
        NodeValue nv;
        memset(nv.data, 0, MAX_VALUE_LEN);
        strncpy(nv.data, v.c_str(), MAX_VALUE_LEN - 1);
        fa_val[idx] = nv;
        fa_left[idx] = NULL_IDX;
        fa_right[idx] = NULL_IDX;
        return idx;
    };

    // Recursive insert into subtree rooted at node_idx
    // Mirrors the private insert(Key, Value, shared_var<node*>&, transaction&)
    std::function<void(int, const std::string &, int)> insert_at;
    insert_at = [&](int key, const std::string &value, int node_idx) {
        if (key < fa_key[node_idx]) {
            int left = fa_left[node_idx];
            if (left != NULL_IDX)
                insert_at(key, value, left);
            else
                fa_left[node_idx] = alloc_node(key, value);
        } else {
            // key >= current node's key: go right (matches original behavior)
            int right = fa_right[node_idx];
            if (right != NULL_IDX)
                insert_at(key, value, right);
            else
                fa_right[node_idx] = alloc_node(key, value);
        }
    };

    // Public insert: mirrors binary_tree::insert(Key, Value const&)
    // The ASTM transaction/commit loop is replaced by exclusive region access
    auto insert = [&](int key, const std::string &value) {
        if (root_idx == NULL_IDX)
            root_idx = alloc_node(key, value);
        else
            insert_at(key, value, root_idx);
    };

    // Recursive search: mirrors binary_tree::search(Key, shared_var<node*>&, transaction&)
    // Returns node index or NULL_IDX if not found
    std::function<int(int, int)> search_at;
    search_at = [&](int key, int node_idx) -> int {
        if (node_idx == NULL_IDX)
            return NULL_IDX;
        int nk = fa_key[node_idx];
        if (key == nk)
            return node_idx;
        if (key < nk)
            return search_at(key, fa_left[node_idx]);
        else
            return search_at(key, fa_right[node_idx]);
    };

    // Public search: mirrors binary_tree::search(Key)
    auto search = [&](int key) -> int {
        return search_at(key, root_idx);
    };

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
        outfile << "  Insert(Key: " << entry.id << ", Value: \"" << entry.val << "\")" << std::endl;
        insert(entry.id, entry.val);
    }
    outfile << "\n";

    // 2. Search Test Phase
    outfile << "Running Search Tests:\n";

    auto perform_search = [&](int search_key) {
        outfile << "  Search(Key: " << search_key << ") -> ";
        int idx = search(search_key);
        if (idx != NULL_IDX) {
            NodeValue nv = fa_val[idx];
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
