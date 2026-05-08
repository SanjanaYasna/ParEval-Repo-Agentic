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

using namespace Legion;

// ---- Enums for task and field IDs ----
enum TaskIDs {
    TOP_LEVEL_TASK_ID,
};

enum FieldIDs {
    FID_KEY,
    FID_VALUE,
    FID_LEFT,
    FID_RIGHT,
};

// Sentinel for null tree pointers (no child)
static const int NULL_NODE = -1;

// Fixed-size string to store in a Legion region field
static const size_t MAX_VALUE_LEN = 64;

struct FixedString {
    char data[MAX_VALUE_LEN];

    FixedString() { memset(data, 0, sizeof(data)); }

    FixedString(const char* s) {
        memset(data, 0, sizeof(data));
        strncpy(data, s, MAX_VALUE_LEN - 1);
    }

    FixedString(const std::string& s) : FixedString(s.c_str()) {}

    std::string str() const { return std::string(data); }
};

// ---- Binary tree operations on region accessors ----

// Insert a (key, value) pair into the tree stored in region accessors.
// Returns the root index (may change if the tree was empty).
int tree_insert(
    const FieldAccessor<READ_WRITE, int, 1>& key_acc,
    const FieldAccessor<READ_WRITE, FixedString, 1>& val_acc,
    const FieldAccessor<READ_WRITE, int, 1>& left_acc,
    const FieldAccessor<READ_WRITE, int, 1>& right_acc,
    int root_idx, int& next_free, int new_key, const FixedString& new_value)
{
    if (root_idx == NULL_NODE) {
        int idx = next_free++;
        key_acc[idx] = new_key;
        val_acc[idx] = new_value;
        left_acc[idx] = NULL_NODE;
        right_acc[idx] = NULL_NODE;
        return idx;
    }

    int node_key = key_acc[root_idx];
    if (new_key < node_key) {
        int left = left_acc[root_idx];
        left = tree_insert(key_acc, val_acc, left_acc, right_acc,
                           left, next_free, new_key, new_value);
        left_acc[root_idx] = left;
    } else {
        // key >= node_key goes right (matches original code)
        int right = right_acc[root_idx];
        right = tree_insert(key_acc, val_acc, left_acc, right_acc,
                            right, next_free, new_key, new_value);
        right_acc[root_idx] = right;
    }
    return root_idx;
}

// Search for a key in the tree. Returns the node index, or NULL_NODE if not found.
int tree_search(
    const FieldAccessor<READ_WRITE, int, 1>& key_acc,
    const FieldAccessor<READ_WRITE, int, 1>& left_acc,
    const FieldAccessor<READ_WRITE, int, 1>& right_acc,
    int root_idx, int search_key)
{
    if (root_idx == NULL_NODE)
        return NULL_NODE;

    int node_key = key_acc[root_idx];
    if (search_key == node_key)
        return root_idx;
    else if (search_key < node_key)
        return tree_search(key_acc, left_acc, right_acc,
                           (int)left_acc[root_idx], search_key);
    else
        return tree_search(key_acc, left_acc, right_acc,
                           (int)right_acc[root_idx], search_key);
}

// ---- Top-level Legion task ----
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

    // Create a logical region to hold tree nodes.
    // Pre-allocate enough slots for all nodes we will insert.
    const size_t max_nodes = 64;

    Rect<1> rect(0, max_nodes - 1);
    IndexSpace is = runtime->create_index_space(ctx, rect);
    FieldSpace fs = runtime->create_field_space(ctx);
    {
        FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
        allocator.allocate_field(sizeof(int),         FID_KEY);
        allocator.allocate_field(sizeof(FixedString), FID_VALUE);
        allocator.allocate_field(sizeof(int),         FID_LEFT);
        allocator.allocate_field(sizeof(int),         FID_RIGHT);
    }
    LogicalRegion tree_lr = runtime->create_logical_region(ctx, is, fs);

    // Inline-map the region with READ_WRITE privilege
    RegionRequirement req(tree_lr, READ_WRITE, EXCLUSIVE, tree_lr);
    req.add_field(FID_KEY);
    req.add_field(FID_VALUE);
    req.add_field(FID_LEFT);
    req.add_field(FID_RIGHT);
    InlineLauncher il(req);
    PhysicalRegion tree_pr = runtime->map_region(ctx, il);
    tree_pr.wait_until_valid();

    // Create field accessors
    const FieldAccessor<READ_WRITE, int, 1>         key_acc(tree_pr,   FID_KEY);
    const FieldAccessor<READ_WRITE, FixedString, 1> val_acc(tree_pr,   FID_VALUE);
    const FieldAccessor<READ_WRITE, int, 1>         left_acc(tree_pr,  FID_LEFT);
    const FieldAccessor<READ_WRITE, int, 1>         right_acc(tree_pr, FID_RIGHT);

    // Initialize all node slots
    for (size_t i = 0; i < max_nodes; i++) {
        key_acc[i]   = 0;
        val_acc[i]   = FixedString();
        left_acc[i]  = NULL_NODE;
        right_acc[i] = NULL_NODE;
    }

    int root_idx = NULL_NODE;
    int next_free = 0;

    // Insert data entries into the tree
    outfile << "Inserting Data:\n";
    for (const auto& entry : data) {
        outfile << "  Insert(Key: " << entry.id
                << ", Value: \"" << entry.val << "\")" << std::endl;
        root_idx = tree_insert(key_acc, val_acc, left_acc, right_acc,
                               root_idx, next_free, entry.id,
                               FixedString(entry.val));
    }
    outfile << "\n";

    // 2. Search Test Phase
    outfile << "Running Search Tests:\n";

    auto perform_search = [&](int search_key) {
        outfile << "  Search(Key: " << search_key << ") -> ";

        int result = tree_search(key_acc, left_acc, right_acc,
                                 root_idx, search_key);

        if (result != NULL_NODE) {
            FixedString found_val = val_acc[result];
            outfile << "Found! Result Value: \"" << found_val.str() << "\"\n";
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

    // Cleanup Legion resources
    runtime->unmap_region(ctx, tree_pr);
    runtime->destroy_logical_region(ctx, tree_lr);
    runtime->destroy_field_space(ctx, fs);
    runtime->destroy_index_space(ctx, is);

    outfile.close();
}

// ---- Main: register tasks and start the Legion runtime ----
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
