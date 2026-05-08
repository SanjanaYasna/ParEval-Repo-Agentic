////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <legion.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <string>
#include <cassert>

using namespace Legion;

static const int NULL_NODE = -1;
static const int MAX_VALUE_LEN = 64;
static const int MAX_NODES = 128;

enum FieldIDs {
    FID_KEY = 100,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
};

enum TaskIDs {
    TOP_LEVEL_TASK_ID = 0,
};

// Fixed-size string wrapper suitable for use as a Legion field type.
struct ValueStr {
    char data[MAX_VALUE_LEN];
    ValueStr() { memset(data, 0, sizeof(data)); }
    ValueStr(const std::string& s) {
        memset(data, 0, sizeof(data));
        strncpy(data, s.c_str(), MAX_VALUE_LEN - 1);
    }
    std::string str() const { return std::string(data); }
};

// Accessor type aliases
typedef FieldAccessor<READ_WRITE, int, 1, coord_t,
    Realm::AffineAccessor<int, 1, coord_t>> RWIntAccessor;
typedef FieldAccessor<READ_WRITE, ValueStr, 1, coord_t,
    Realm::AffineAccessor<ValueStr, 1, coord_t>> RWValueAccessor;

// Binary tree state backed by a Legion PhysicalRegion.
// Nodes are stored in the region at consecutive indices; child links are
// integer indices (NULL_NODE for empty).
struct TreeState {
    int root_idx;
    int next_free;

    RWIntAccessor   key_acc;
    RWValueAccessor val_acc;
    RWIntAccessor   left_acc;
    RWIntAccessor   right_acc;

    TreeState(const PhysicalRegion& pr)
      : root_idx(NULL_NODE),
        next_free(0),
        key_acc(pr, FID_KEY),
        val_acc(pr, FID_VALUE),
        left_acc(pr, FID_LEFT),
        right_acc(pr, FID_RIGHT)
    {}

    // Allocate a new node in the region and return its index.
    int alloc_node(int key, const std::string& value) {
        int idx = next_free++;
        assert(idx < MAX_NODES);
        key_acc[idx]   = key;
        val_acc[idx]   = ValueStr(value);
        left_acc[idx]  = NULL_NODE;
        right_acc[idx] = NULL_NODE;
        return idx;
    }

    // Public insert – mirrors the original transactional insert.
    void insert(int key, const std::string& value) {
        if (root_idx == NULL_NODE) {
            root_idx = alloc_node(key, value);
        } else {
            insert_impl(key, value, root_idx);
        }
    }

    // Public search – returns node index or NULL_NODE.
    int search(int key) {
        return search_impl(key, root_idx);
    }

    std::string get_value(int idx) {
        return val_acc[idx].str();
    }

  private:
    void insert_impl(int key, const std::string& value, int node_idx) {
        int node_key = key_acc[node_idx];
        if (key < node_key) {
            int left = left_acc[node_idx];
            if (left != NULL_NODE)
                insert_impl(key, value, left);
            else
                left_acc[node_idx] = alloc_node(key, value);
        } else {
            // key >= node_key goes right (matches original behaviour)
            int right = right_acc[node_idx];
            if (right != NULL_NODE)
                insert_impl(key, value, right);
            else
                right_acc[node_idx] = alloc_node(key, value);
        }
    }

    int search_impl(int key, int node_idx) {
        if (node_idx == NULL_NODE)
            return NULL_NODE;
        int node_key = key_acc[node_idx];
        if (key == node_key)
            return node_idx;
        else if (key < node_key)
            return search_impl(key, left_acc[node_idx]);
        else
            return search_impl(key, right_acc[node_idx]);
    }
};

// -------------------------------------------------------------------
// Top-level Legion task – replaces the original main().
// -------------------------------------------------------------------
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

    // Create index space, field space, and logical region for tree nodes.
    IndexSpace is = runtime->create_index_space(ctx,
        Rect<1>(0, MAX_NODES - 1));
    runtime->attach_name(is, "tree_node_is");

    FieldSpace fs = runtime->create_field_space(ctx);
    runtime->attach_name(fs, "tree_node_fs");
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int),      FID_KEY);
        fa.allocate_field(sizeof(ValueStr), FID_VALUE);
        fa.allocate_field(sizeof(int),      FID_LEFT);
        fa.allocate_field(sizeof(int),      FID_RIGHT);
    }

    LogicalRegion tree_lr = runtime->create_logical_region(ctx, is, fs);
    runtime->attach_name(tree_lr, "tree_nodes");

    // Inline-map the region with READ_WRITE privileges so we can
    // operate on it directly (analogous to the single-threaded
    // transactional access in the original code).
    InlineLauncher il(RegionRequirement(tree_lr, READ_WRITE,
                                        EXCLUSIVE, tree_lr));
    il.add_field(FID_KEY);
    il.add_field(FID_VALUE);
    il.add_field(FID_LEFT);
    il.add_field(FID_RIGHT);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();

    {
        TreeState tree(pr);

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
            tree.insert(entry.id, entry.val);
        }
        outfile << "\n";

        // 2. Search Test Phase
        outfile << "Running Search Tests:\n";

        auto perform_search = [&](int search_key) {
            outfile << "  Search(Key: " << search_key << ") -> ";

            int idx = tree.search(search_key);

            if (idx != NULL_NODE) {
                outfile << "Found! Result Value: \""
                        << tree.get_value(idx) << "\"\n";
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

    // Clean up Legion resources.
    runtime->unmap_region(ctx, pr);
    runtime->destroy_logical_region(ctx, tree_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}

// -------------------------------------------------------------------
// main – register the top-level task and start the Legion runtime.
// -------------------------------------------------------------------
int main(int argc, char **argv)
{
    Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

    {
        TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<top_level_task>(registrar,
                                                         "top_level");
    }

    return Runtime::start(argc, argv);
}
