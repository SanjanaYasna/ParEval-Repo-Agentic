////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2014 Steve Brandt
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Translated from HPX/ASTM to the Legion execution model.
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include "legion.h"

using namespace Legion;

// ---- Constants and enums ----

enum TaskIDs {
    TOP_LEVEL_TASK_ID,
};

enum FieldIDs {
    FID_KEY,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
};

static const int MAX_NODES = 128;
static const int NULL_NODE = -1;
static const int MAX_VALUE_LEN = 64;

// Fixed-size string wrapper suitable for storage in a Legion region field
struct FixedString {
    char data[MAX_VALUE_LEN];

    FixedString() { memset(data, 0, sizeof(data)); }

    FixedString(const std::string& s) {
        memset(data, 0, sizeof(data));
        strncpy(data, s.c_str(), MAX_VALUE_LEN - 1);
    }

    FixedString(const char* s) {
        memset(data, 0, sizeof(data));
        if (s) strncpy(data, s, MAX_VALUE_LEN - 1);
    }

    std::string str() const { return std::string(data); }
};

// ---- BinaryTree operating on Legion region accessors ----
//
// Replaces the ASTM shared_var<T> / transaction model:
//   - shared_var<Key>, shared_var<node*> fields  →  region fields accessed via FieldAccessor
//   - transaction + commit_transaction()          →  region privilege (READ_WRITE, EXCLUSIVE)
//   - new node(key, value)                        →  allocate_node() that writes into the region

class BinaryTree {
public:
    BinaryTree(PhysicalRegion pr)
        : acc_key(pr, FID_KEY),
          acc_value(pr, FID_VALUE),
          acc_left(pr, FID_LEFT),
          acc_right(pr, FID_RIGHT),
          root_idx(NULL_NODE),
          next_free(0)
    {}

    void insert(int key, const std::string& value)
    {
        if (root_idx == NULL_NODE) {
            root_idx = allocate_node(key, value);
        } else {
            insert_at(key, value, root_idx);
        }
    }

    // Returns node index, or NULL_NODE if not found
    int search(int key)
    {
        return search_at(key, root_idx);
    }

    int         get_key(int idx)   const { return acc_key[idx]; }
    std::string get_value(int idx) const {
        FixedString fs = acc_value[idx];
        return fs.str();
    }

private:
    int allocate_node(int key, const std::string& value)
    {
        int idx = next_free++;
        acc_key[idx]   = key;
        acc_value[idx] = FixedString(value);
        acc_left[idx]  = NULL_NODE;
        acc_right[idx] = NULL_NODE;
        return idx;
    }

    // Mirrors the original recursive insert with key < / key >= branching
    void insert_at(int key, const std::string& value, int node_idx)
    {
        if (key < acc_key[node_idx]) {
            int left = acc_left[node_idx];
            if (left != NULL_NODE)
                insert_at(key, value, left);
            else
                acc_left[node_idx] = allocate_node(key, value);
        } else { // key >= node key  (same semantics as original)
            int right = acc_right[node_idx];
            if (right != NULL_NODE)
                insert_at(key, value, right);
            else
                acc_right[node_idx] = allocate_node(key, value);
        }
    }

    int search_at(int key, int node_idx)
    {
        if (node_idx == NULL_NODE)
            return NULL_NODE;

        int node_key = acc_key[node_idx];
        if (key == node_key)
            return node_idx;
        else if (key < node_key)
            return search_at(key, acc_left[node_idx]);
        else
            return search_at(key, acc_right[node_idx]);
    }

    // Region field accessors (replace shared_var reads/writes inside transactions)
    const FieldAccessor<READ_WRITE, int, 1>         acc_key;
    const FieldAccessor<READ_WRITE, FixedString, 1>  acc_value;
    const FieldAccessor<READ_WRITE, int, 1>         acc_left;
    const FieldAccessor<READ_WRITE, int, 1>         acc_right;

    int root_idx;   // index of root node (replaces shared_var<node*> root)
    int next_free;  // next free slot in the region
};

// ---- Top-level Legion task (replaces main + hpx_main) ----

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

    // ----- Create a logical region to hold tree nodes -----
    Rect<1> node_rect(0, MAX_NODES - 1);
    IndexSpaceT<1> is = runtime->create_index_space(ctx, node_rect);
    runtime->attach_name(is, "tree_node_is");

    FieldSpace fs = runtime->create_field_space(ctx);
    runtime->attach_name(fs, "tree_node_fs");
    {
        FieldAllocator fa = runtime->create_field_allocator(ctx, fs);
        fa.allocate_field(sizeof(int),         FID_KEY);
        fa.allocate_field(sizeof(FixedString), FID_VALUE);
        fa.allocate_field(sizeof(int),         FID_LEFT);
        fa.allocate_field(sizeof(int),         FID_RIGHT);
    }

    LogicalRegion tree_lr = runtime->create_logical_region(ctx, is, fs);
    runtime->attach_name(tree_lr, "tree_nodes");

    // Inline-map the region with READ_WRITE / EXCLUSIVE privileges.
    // This is the Legion equivalent of the ASTM transaction scope that
    // wrapped every insert / search in the original code.
    InlineLauncher il(RegionRequirement(tree_lr, READ_WRITE, EXCLUSIVE, tree_lr));
    il.add_field(FID_KEY);
    il.add_field(FID_VALUE);
    il.add_field(FID_LEFT);
    il.add_field(FID_RIGHT);
    PhysicalRegion pr = runtime->map_region(ctx, il);
    pr.wait_until_valid();

    // Build the tree on top of the mapped region
    {
        BinaryTree tree(pr);

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
                outfile << "Found! Result Value: \"" << tree.get_value(idx) << "\"\n";
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

    // Clean up Legion resources
    runtime->unmap_region(ctx, pr);
    runtime->destroy_logical_region(ctx, tree_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);
}

// ---- Entry point ----

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
